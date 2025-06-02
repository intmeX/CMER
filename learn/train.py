import shutil
import logging
import time

import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ConstantLR, ExponentialLR, CosineAnnealingLR, CyclicLR, LinearLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from data.emotic_dataset import Emotic_PreDataset, EmoticTripleStreamDataset
from data.caer_dataset import CAERDataset
from model import DiscreteLoss, ContinuousLoss_SL1, ContinuousLoss_L2, DiceLoss, ZLPRLoss, BCEDiscreteLoss, \
    FocalDiscreteLoss
from model import prep_models_double_stream, prep_models_triple_stream, prep_models_quadruple_stream, \
    prep_models_single_face
from model import prep_models_multistream, prep_models_caer_multistream
from learn.test import test_vad_mse, test_scikit_ap, test_vad, test_emotic


def train_epoch(model, criteria, optimizer, scheduler, warmup_scheduler, device, e, loader, writer, args):
    logger = logging.getLogger('Experiment')
    # train models for one epoch
    iters = e * len(loader)
    num_samples = 0
    running_loss = 0.0
    running_cat_loss = 0.0
    running_cont_loss = 0.0

    model.train()

    disc_loss, cont_loss = criteria
    for batch in iter(loader):
        labels_cat = batch[-2].to(device)
        labels_cont = batch[-1].to(device)
        imgs = []
        for img in batch[:-2]:
            imgs.append(img.to(device))

        optimizer.zero_grad()

        pred_cat, pred_cont = model(*imgs)
        cat_loss_batch = disc_loss(pred_cat, labels_cat)
        cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)

        loss = (args.cat_loss_weight * cat_loss_batch) + (args.cont_loss_weight * cont_loss_batch)

        running_loss += loss.item()
        running_cat_loss += cat_loss_batch.item()
        running_cont_loss += cont_loss_batch.item()

        loss.backward()
        optimizer.step()
        num_samples += labels_cat.shape[0]
        iters += 1
        if iters % 10 == 0:
            writer.add_scalar('train_losses/total_loss', running_loss / num_samples, iters)
            writer.add_scalar('train_losses/categorical_loss', running_cat_loss / num_samples, iters)
            writer.add_scalar('train_losses/continuous_loss', running_cont_loss / num_samples, iters)
            writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], iters)
            logger.info(
                'epoch[{}]: step = {} lr = {:.6f} total_loss = {:.4f} categorical_loss = {:.4f} continuous_loss = {:.4f}'.format(
                    e, iters, optimizer.param_groups[0]['lr'], running_loss / num_samples,
                                                               running_cat_loss / num_samples,
                                                               running_cont_loss / num_samples))
        if iters <= args.warmup:
            warmup_scheduler.step()
        elif iters > args.decay_start:
            scheduler.step()
        else:
            pass

    logger.info('epoch = %d loss = %.4f cat loss = %.4f cont_loss = %.4f\n' % (
        e, running_loss / num_samples, running_cat_loss / num_samples, running_cont_loss / num_samples))


def val_epoch(model, criteria, device, e, loader, writer, args):
    logger = logging.getLogger('Experiment')
    # validation for one epoch
    iters = e * len(loader)
    num_samples = 0
    running_loss = 0.0
    running_cat_loss = 0.0
    running_cont_loss = 0.0
    indx = 0
    num_images = len(loader.dataset)
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))
    cont_preds = np.zeros((num_images, 3))
    cont_labels = np.zeros((num_images, 3))

    model.eval()

    disc_loss, cont_loss = criteria
    with torch.no_grad():
        for batch in iter(loader):
            labels_cat = batch[-2].to(device)
            labels_cont = batch[-1].to(device)
            imgs = []
            for img in batch[:-2]:
                imgs.append(img.to(device))

            pred_cat, pred_cont = model(*imgs)
            cat_loss_batch = disc_loss(pred_cat, labels_cat)
            cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
            loss = (args.cat_loss_weight * cat_loss_batch) + (args.cont_loss_weight * cont_loss_batch)

            running_loss += loss.item()
            running_cat_loss += cat_loss_batch.item()
            running_cont_loss += cont_loss_batch.item()
            num_samples += labels_cat.shape[0]
            iters += 1
            if iters % 10 == 0:
                writer.add_scalar('val_losses/total_loss', running_loss / num_samples, iters)
                writer.add_scalar('val_losses/categorical_loss', running_cat_loss / num_samples, iters)
                writer.add_scalar('val_losses/continuous_loss', running_cont_loss / num_samples, iters)
                logger.info(
                    'val epoch[{}]: step = {} total_loss = {:.4f} categorical_loss = {:.4f} continuous_loss = {:.4f}'.format(
                        e, iters, running_loss / num_samples, running_cat_loss / num_samples,
                                  running_cont_loss / num_samples))

            cat_preds[indx: (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[indx: (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            cont_preds[indx: (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
            cont_labels[indx: (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
            indx = indx + pred_cat.shape[0]

    logger.info('val epoch = %d loss = %.4f cat loss = %.4f cont_loss = %.4f\n' % (
        e, running_loss / num_samples, running_cat_loss / num_samples, running_cont_loss / num_samples))

    return (
        cat_preds.transpose(),
        cat_labels.transpose(),
        cont_preds.transpose(),
        cont_labels.transpose(),
    )


def save_checkpoint(ckpt, model_path, is_best=False):
    os.system('rm -rf {}/checkpoint_{}*'.format(model_path, ckpt['brief']))
    filename = '{}/checkpoint_{}_epoch{}.pth'.format(model_path, ckpt['brief'], ckpt['epoch'])
    torch.save(ckpt, filename)
    if is_best:
        shutil.copyfile(filename, '{}/best_checkpoint.pth'.format(model_path))


def train_data(optimizer, scheduler, warmup_scheduler, model, device, train_loader, val_loader, criteria,
               writer, model_path, ind2cat, ind2vad, args):
    '''
    Training emotic model on train data using train loader.
    :param opt: Optimizer object.
    :param scheduler: Learning rate scheduler object.
    :param model: emotic_model (fusion model).
    :param device: Torch device. Used to send tensors to GPU if available. 
    :param train_loader: Dataloader iterating over train dataset. 
    :param val_loader: Dataloader iterating over validation dataset.
    :param criteria: list of Discrete loss criterion and continuous loss criterion
    :param writer: SummaryWriter object to save logs.
    # :param val_writer: SummaryWriter object to save validation logs.
    :param model_path: Directory path to save the models after training. 
    :param args: Runtime arguments.
    '''

    logger = logging.getLogger('Experiment')
    model.to(device)
    logger.info('starting training')

    best_mAP = 0
    for e in range(args.epochs):
        start_time = time.time()
        train_epoch(model, criteria, optimizer, scheduler, warmup_scheduler, device, e, train_loader, writer, args)
        cat_preds, cat_labels, cont_preds, cont_labels = val_epoch(model, criteria, device, e, val_loader, writer, args)

        ap = test_scikit_ap(cat_preds, cat_labels, ind2cat, output=True)
        vad_mse = test_vad_mse(cont_preds, cont_labels, ind2vad, output=True)
        vad = test_vad(cont_preds, cont_labels, ind2vad, output=True)
        mAP = ap.mean()

        writer.add_scalar('mAP/0_total_mAP', mAP, e + 1)
        for i in range(26):
            writer.add_scalar('mAP/{}'.format(ind2cat[i]), ap[i], e + 1)
        writer.add_scalar('vad_mse/0_total_mse', vad_mse.mean(), e + 1)
        for i in range(3):
            writer.add_scalar('vad_mse/{}'.format(ind2vad[i]), vad_mse[i], e + 1)
        writer.add_scalar('vad/0_total', vad.mean(), e + 1)
        for i in range(3):
            writer.add_scalar('vad/{}'.format(ind2vad[i]), vad[i], e + 1)
        # logger.info('epoch[{}] time = {:.2f}s'.format(e, time.time() - start_time))

        # scheduler.step()

        is_best = False
        if mAP > best_mAP:
            is_best = True
            best_mAP = mAP
        save_checkpoint(
            {
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "best_mAP": best_mAP,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "warmup_scheduler": warmup_scheduler.state_dict(),
                "brief": model.brief,
            },
            model_path,
            is_best,
        )

        logger.info('epoch[{}] time = {:.2f}s maxAP = {:.6f}'.format(e, time.time() - start_time, best_mAP))
        all_log_dir = os.path.join('./all_logs', args.experiment_id + args.experiment_name)
        shutil.copytree(writer.logdir, all_log_dir, dirs_exist_ok=True)

    logger.info('completed training')


def train_emotic(result_path, model_path, train_log_path, context_norm, body_norm, face_norm, args):
    ''' Prepare dataset, dataloders, models. 
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param train_log_path: Directory path to save the training logs.
    # :param val_log_path: Directoty path to save the validation logs.
    :param context_norm: List containing mean and std values for context images.
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments. 
    '''
    logger = logging.getLogger('Experiment')
    # torch.autograd.set_detect_anomaly(True)
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness',
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion
    vad = ['Valence', 'Arousal', 'Dominance']
    ind2vad = {}
    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous
    # Load preprocessed data from npy files
    if args.context_mask:
        train_context = np.load(os.path.join(args.data_path, 'train_mask_context_arr.npy'))
        val_context = np.load(os.path.join(args.data_path, 'val_mask_context_arr.npy'))
        test_context = np.load(os.path.join(args.data_path, 'test_mask_context_arr.npy'))
    else:
        train_context = np.load(os.path.join(args.data_path, 'train_context_arr.npy'))
        val_context = np.load(os.path.join(args.data_path, 'val_context_arr.npy'))
        test_context = np.load(os.path.join(args.data_path, 'test_context_arr.npy'))

    train_body = np.load(os.path.join(args.data_path, 'train_body_arr.npy'))
    train_cat = np.load(os.path.join(args.data_path, 'train_cat_arr.npy'))
    train_cont = np.load(os.path.join(args.data_path, 'train_cont_arr.npy'))
    train_face = np.stack((np.load(os.path.join(args.data_path, 'train_face_arr.npy')),) * 3, axis=-1)

    val_body = np.load(os.path.join(args.data_path, 'val_body_arr.npy'))
    val_cat = np.load(os.path.join(args.data_path, 'val_cat_arr.npy'))
    val_cont = np.load(os.path.join(args.data_path, 'val_cont_arr.npy'))
    val_face = np.stack((np.load(os.path.join(args.data_path, 'val_face_arr.npy')),) * 3, axis=-1)

    test_body = np.load(os.path.join(args.data_path, 'test_body_arr.npy'))
    test_cat = np.load(os.path.join(args.data_path, 'test_cat_arr.npy'))
    test_cont = np.load(os.path.join(args.data_path, 'test_cont_arr.npy'))
    test_face = np.stack((np.load(os.path.join(args.data_path, 'test_face_arr.npy')),) * 3, axis=-1)
    logger.info('test  context ' + str(test_context.shape) + ' body ' + str(test_body.shape) + ' face ' +
                str(test_face.shape) + ' cat ' + str(test_cat.shape) + ' cont ' + str(test_cont.shape))

    print('train ', 'context ', train_context.shape, 'body', train_body.shape, 'face', train_face.shape, 'cat ',
          train_cat.shape, 'cont', train_cont.shape)
    print('val ', 'context ', val_context.shape, 'body', val_body.shape, 'face', val_body.shape, 'cat ', val_cat.shape,
          'cont', val_cont.shape)

    # Initialize Dataset and DataLoader 
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    if args.body_model == 'swin_t':
        train_transform = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(size=[232],
                                                                interpolation=transforms.InterpolationMode.BICUBIC),
                                              transforms.CenterCrop(size=[224]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                              transforms.ToTensor()])

        test_transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(size=[232],
                                                               interpolation=transforms.InterpolationMode.BICUBIC),
                                             transforms.CenterCrop(size=[224]),
                                             transforms.ToTensor()])

    face_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
    ])
    face_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    if args.arch == 'double_stream':
        train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform,
                                          context_norm, body_norm)
        val_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform, context_norm,
                                        body_norm)
        emotic_model = prep_models_double_stream(context_model=args.context_model, body_model=args.body_model,
                                                 args=args)
    else:
        train_dataset = EmoticTripleStreamDataset(train_context, train_body, train_face, train_cat, train_cont,
                                                  train_transform, train_transform, face_train_transform,
                                                  context_norm, body_norm, face_norm)
        val_dataset = EmoticTripleStreamDataset(val_context, val_body, val_face, val_cat, val_cont,
                                                test_transform, test_transform, face_test_transform,
                                                context_norm, body_norm, face_norm)
        if args.arch == 'triple_stream':
            emotic_model = prep_models_triple_stream(context_model=args.context_model, body_model=args.body_model,
                                                     face_model=args.face_model, args=args)
        elif args.arch == 'single_face':
            emotic_model = prep_models_single_face(face_model=args.face_model, args=args)
        elif args.arch == 'quadruple_stream':
            emotic_model = prep_models_quadruple_stream(context_model=args.context_model, body_model=args.body_model,
                                                        face_model=args.face_model, caption_model=args.caption_model,
                                                        args=args)
        else:
            emotic_model = prep_models_multistream(context_model=args.context_model, body_model=args.body_model,
                                                   face_model=args.face_model, caption_model=args.caption_model,
                                                   args=args)

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed),
                              num_workers=args.num_worker, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                            num_workers=args.num_worker, pin_memory=True)

    print('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # Prepare models 
    # emotic_model = prep_models(context_model=args.context_model, body_model=args.body_model, model_dir=model_path)

    # for param in emotic_model.parameters():
    #     param.requires_grad = True

    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    # emotic_model = emotic_model.to(device)
    logger.info('device: {}'.format(str(device)))
    if args.optimizer == 'SGD':
        optimizer = optim.SGD((list(emotic_model.parameters())), lr=args.learning_rate, weight_decay=args.weight_decay,
                              momentum=args.sgd_momentum)
    else:
        optimizer = optim.Adam((list(emotic_model.parameters())), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == 'const':
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
    elif args.scheduler == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    else:  # args.scheduler == 'cosine':
        # half cycle
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs - args.decay_start)
    start_factor = 1.0
    if args.warmup > 0:
        start_factor = 0.01
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=args.warmup)

    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    if args.discrete_loss_type == 'default':
        disc_loss = DiscreteLoss(args.discrete_loss_weight_type, device)
    elif args.discrete_loss_type == 'bce':
        pos_num = torch.tensor(train_cat.sum(axis=0), dtype=torch.float32)
        pos_weight = (train_cat.shape[0] - pos_num + 1.0) / (pos_num + 1.0)
        logger.info('the neg/pos of dataset:')
        logger.info(pos_weight)
        disc_loss = BCEDiscreteLoss(weight_type=args.discrete_loss_weight_type, static_pos_weights=pos_weight,
                                    device=device)
    elif args.discrete_loss_type == 'dice':
        disc_loss = DiceLoss()
    elif args.discrete_loss_type == 'focal':
        pos_num = torch.tensor(train_cat.sum(axis=0), dtype=torch.float32)
        pos_weight = (train_cat.shape[0] - pos_num + 1.0) / (pos_num + 1.0)
        logger.info('the neg/pos of dataset:')
        logger.info(pos_weight)
        disc_loss = FocalDiscreteLoss(weight_type=args.discrete_loss_weight_type, static_pos_weights=pos_weight,
                                      hard_gamma=args.hard_gamma, device=device)
    else:
        disc_loss = ZLPRLoss()
    if args.continuous_loss_type == 'Smooth L1':
        cont_loss = ContinuousLoss_SL1()
    else:
        cont_loss = ContinuousLoss_L2()
    criteria = [
        disc_loss,
        cont_loss,
    ]

    writer = SummaryWriter(train_log_path)

    # training
    train_data(optimizer, scheduler, warmup_scheduler, emotic_model, device, train_loader, val_loader,
               criteria, writer, model_path, ind2cat, ind2vad, args)


def train_epoch_caer(model, criteria, optimizer, scheduler, warmup_scheduler, device, e, loader, writer, args):
    logger = logging.getLogger('Experiment')
    # train models for one epoch
    iters = e * len(loader)
    num_samples = 0
    running_loss = 0.0
    cor = 0

    model.train()

    disc_loss, cont_loss = criteria
    for idx, batch in enumerate(loader):
        labels_cat = batch[-1].to(device).long()
        imgs = []
        for img in batch[:-1]:
            imgs.append(img.to(device))

        optimizer.zero_grad()

        pred_cat, pred_cont = model(*imgs)
        loss = disc_loss(pred_cat, labels_cat)
        pred = pred_cat.argmax(dim=1)
        cor += torch.sum(pred == labels_cat)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        num_samples += labels_cat.shape[0]
        iters += 1
        if iters % 10 == 0:
            writer.add_scalar('train_losses/total_loss', running_loss / (idx + 1), iters)
            writer.add_scalar('train_lr', optimizer.param_groups[0]['lr'], iters)
            logger.info(
                'epoch[{}]: step = {} lr = {:.6f} total_loss = {:.4f}'.format(
                    e, iters, optimizer.param_groups[0]['lr'], running_loss / (idx + 1),
                ))
        if iters <= args.warmup:
            warmup_scheduler.step()
        elif iters > args.decay_start:
            scheduler.step()
        else:
            pass
    logger.info('epoch = %d loss = %.4f Acc = %.4f\n' % (e, running_loss / len(loader), cor / num_samples))


def val_epoch_caer(model, criteria, device, e, loader, writer, args):
    logger = logging.getLogger('Experiment')
    # validation for one epoch
    iters = e * len(loader)
    num_samples = 0
    running_loss = 0.0
    cor = 0

    model.eval()
    disc_loss, cont_loss = criteria
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            labels_cat = batch[-1].to(device).long()
            imgs = []
            for img in batch[:-1]:
                imgs.append(img.to(device))
            pred_cat, pred_cont = model(*imgs)
            loss = disc_loss(pred_cat, labels_cat)
            pred = pred_cat.argmax(dim=1)
            cor += torch.sum(pred == labels_cat)
            running_loss += loss.item()
            num_samples += labels_cat.shape[0]
            iters += 1
            if iters % 10 == 0:
                writer.add_scalar('val_losses/total_loss', running_loss / (idx + 1), iters)
                logger.info('val epoch[{}]: step = {} total_loss = {:.4f}'.format(e, iters, running_loss / (idx + 1)))
    logger.info('val epoch = %d loss = %.4f Acc = %.4f\n' % (e, running_loss / len(loader), cor / num_samples))
    return cor / num_samples


def train_data_caer(optimizer, scheduler, warmup_scheduler, model, device, train_loader, val_loader, criteria, writer,
                    model_path, args):
    '''
    Training emotic model on train data using train loader.
    :param opt: Optimizer object.
    :param scheduler: Learning rate scheduler object.
    :param model: emotic_model (fusion model).
    :param device: Torch device. Used to send tensors to GPU if available.
    :param train_loader: Dataloader iterating over train dataset.
    :param val_loader: Dataloader iterating over validation dataset.
    :param criteria: list of Discrete loss criterion and continuous loss criterion
    :param writer: SummaryWriter object to save logs.
    # :param val_writer: SummaryWriter object to save validation logs.
    :param model_path: Directory path to save the models after training.
    :param args: Runtime arguments.
    '''

    logger = logging.getLogger('Experiment')
    model.to(device)
    logger.info('starting training')

    best_Acc = 0
    for e in range(args.epochs):
        start_time = time.time()
        train_epoch_caer(model, criteria, optimizer, scheduler, warmup_scheduler, device, e, train_loader, writer, args)
        acc = val_epoch_caer(model, criteria, device, e, val_loader, writer, args)

        writer.add_scalar('mAP/0_total_Acc', acc, e + 1)
        # logger.info('epoch[{}] time = {:.2f}s'.format(e, time.time() - start_time))

        is_best = False
        if acc > best_Acc:
            is_best = True
            best_Acc = acc
        save_checkpoint(
            {
                "epoch": e + 1,
                "state_dict": model.state_dict(),
                "best_Acc": best_Acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "warmup_scheduler": warmup_scheduler.state_dict(),
                "brief": model.brief,
            },
            model_path,
            is_best,
        )
        logger.info('epoch[{}] time = {:.2f}s maxAcc = {:.6f}'.format(e, time.time() - start_time, best_Acc))
        all_log_dir = os.path.join('./all_logs', args.experiment_id + args.experiment_name)
        shutil.copytree(writer.logdir, all_log_dir, dirs_exist_ok=True)
    logger.info('completed training')


def train_caer(result_path, model_path, train_log_path, context_norm, body_norm, face_norm, args):
    ''' Prepare dataset, dataloders, models.
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training.
    :param train_log_path: Directory path to save the training logs.
    # :param val_log_path: Directoty path to save the validation logs.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :param context_norm: List containing mean and std values for context images.
    :param body_norm: List containing mean and std values for body images.
    :param args: Runtime arguments.
    '''
    logger = logging.getLogger('Experiment')
    cat = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion
    # torch.autograd.set_detect_anomaly(True)

    # Initialize Dataset and DataLoader
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    if args.body_model == 'swin_t':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=[224]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
        ])

    face_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
    ])
    face_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    if args.arch == 'caer_multistream':
        train_dataset = CAERDataset(args.data_path, os.path.join(args.data_path, 'train.txt'), train_transform,
                                    face_train_transform, context_norm, face_norm)
        val_dataset = CAERDataset(args.data_path, os.path.join(args.data_path, 'test.txt'), test_transform,
                                   face_test_transform, context_norm, face_norm)
        caer_model = prep_models_caer_multistream(context_model=args.context_model, face_model=args.face_model,
                                                  caption_model=args.caption_model, args=args)
    else:
        raise ValueError('No such arch for caer')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=args.num_worker, pin_memory=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,
                            num_workers=args.num_worker, pin_memory=True)

    print('train loader ', len(train_loader), 'val loader ', len(val_loader))

    # Prepare models
    # emotic_model = prep_models(context_model=args.context_model, body_model=args.body_model, model_dir=model_path)

    # for param in emotic_model.parameters():
    #     param.requires_grad = True

    device = torch.device("cuda:%s" % (str(args.gpu)) if torch.cuda.is_available() else "cpu")
    # emotic_model = emotic_model.to(device)
    logger.info('device: {}'.format(str(device)))
    if args.optimizer == 'SGD':
        optimizer = optim.SGD((list(caer_model.parameters())), lr=args.learning_rate, weight_decay=args.weight_decay,
                              momentum=args.sgd_momentum)
    else:
        optimizer = optim.Adam((list(caer_model.parameters())), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == 'const':
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
    elif args.scheduler == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    else:  # args.scheduler == 'cosine':
        # half cycle
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs - args.decay_start)
    start_factor = 1.0
    if args.warmup > 0:
        start_factor = 0.01
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=args.warmup)

    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    if args.discrete_loss_type == 'default':
        disc_loss = DiscreteLoss(args.discrete_loss_weight_type, device)
    elif args.discrete_loss_type == 'ce':
        disc_loss = nn.CrossEntropyLoss()
    else:
        disc_loss = ZLPRLoss()
    if args.continuous_loss_type == 'Smooth L1':
        cont_loss = ContinuousLoss_SL1()
    else:
        cont_loss = ContinuousLoss_L2()
    criteria = [
        disc_loss,
        cont_loss,
    ]

    writer = SummaryWriter(train_log_path)

    # training
    train_data_caer(optimizer, scheduler, warmup_scheduler, caer_model, device, train_loader, val_loader, criteria, writer, model_path, args)
