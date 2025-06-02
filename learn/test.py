import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.io
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.emotic_dataset import Emotic_PreDataset, EmoticTripleStreamDataset
from data.caer_dataset import CAERDataset
from model import prep_models_double_stream, prep_models_triple_stream, prep_models_quadruple_stream, \
    prep_models_single_face, prep_models_multistream, prep_models_caer_multistream


def test_scikit_ap(cat_preds, cat_labels, ind2cat, output=True):
    ''' Calculate average precision per emotion category using sklearn library.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :return: Numpy array containing average precision per emotion category.
    '''
    logger = logging.getLogger('Experiment')
    ap = np.zeros(26, dtype=np.float32)
    for i in range(26):
        ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
        if output:
            logger.info('Category %16s %.5f' %(ind2cat[i], ap[i]))
            # print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
    if output:
        logger.info('Mean AP %.5f' %(ap.mean()))
        # print ('Mean AP %.5f' %(ap.mean()))
    return ap


def test_vad(cont_preds, cont_labels, ind2vad, output=True):
    ''' Calcaulate VAD (valence, arousal, dominance) errors.
    :param cont_preds: Continuous emotion predictions.
    :param cont_labels: Continuous emotion labels.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :return: Numpy array containing mean absolute error per continuous emotion dimension.
    '''
    logger = logging.getLogger('Experiment')
    vad = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
        if output:
            logger.info('Continuous %10s %.5f' % (ind2vad[i], vad[i]))
            # print ('Continuous %10s %.5f' %(ind2vad[i], vad[i]))
    if output:
        logger.info('Mean VAD Error %.5f' %(vad.mean()))
        # print ('Mean VAD Error %.5f' %(vad.mean()))
    return vad


def test_vad_mse(cont_preds, cont_labels, ind2vad, output=True):
    ''' Calcaulate VAD (valence, arousal, dominance) mse errors.
    :param cont_preds: Continuous emotion predictions.
    :param cont_labels: Continuous emotion labels.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
    :return: Numpy array containing mean mse error per continuous emotion dimension.
    '''
    logger = logging.getLogger('Experiment')
    vad_mse = np.zeros(3, dtype=np.float32)
    for i in range(3):
        vad_mse[i] = np.mean((cont_preds[i, :] - cont_labels[i, :]) ** 2)
        if output:
            logger.info('Continuous %10s MSE: %.5f' %(ind2vad[i], vad_mse[i]))
            # print ('Continuous %10s MSE: %.5f' %(ind2vad[i], vad_mse[i]))
    if output:
        logger.info('Mean VAD MSE Error %.5f' %(vad_mse.mean()))
        # print ('Mean VAD MSE Error %.5f' %(vad_mse.mean()))
    return vad_mse


def get_thresholds(cat_preds, cat_labels):
    ''' Calculate thresholds where precision is equal to recall. These thresholds are then later for inference.
    :param cat_preds: Categorical emotion predictions.
    :param cat_labels: Categorical emotion labels.
    :return: Numpy array containing thresholds per emotion category where precision is equal to recall.
    '''
    thresholds = np.zeros(26, dtype=np.float32)
    for i in range(26):
        p, r, t = precision_recall_curve(cat_labels[i, :], cat_preds[i, :])
        for k in range(len(p)):
            if p[k] == r[k]:
                thresholds[i] = t[k]
                break
    return thresholds


def calc_confusion(cat_pred, cat_labels, ind2cat, thresholds, save_path='./', num_classes=26):
    cm = [np.zeros((2, 2), dtype=np.float32) for i in range(num_classes)]
    preds = np.array([cat_pred[i] > thresholds[i] for i in range(num_classes)])
    for i in range(num_classes):
        true_ind = cat_labels[i] > 0.5
        pos_num = np.sum(true_ind)
        hit_ratio = np.sum(preds[i][true_ind]) / pos_num
        cm[i] = hit_ratio
    cat_list = [ind2cat[i] for i in range(num_classes)]
    cm = pd.DataFrame(cm, columns=cat_list, index=cat_list)
    ax = sns.heatmap(cm, annot=True, fmt='d')
    hm = ax.get_figure()
    hm.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=400)


def test_data(model, device, loader, ind2cat, ind2vad, num_images, result_dir='./', test_type='val'):
    ''' Test models on data 
    :param models: List containing model_context, model_body and emotic_model (fusion model) in that order.
    :param device: Torch device. Used to send tensors to GPU if available. 
    :param loader: Dataloader iterating over dataset.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance)
    :param num_images: Number of images in the dataset. 
    :param result_dir: Directory path to save results (predictions mat object and thresholds npy object).
    :param test_type: Test type variable. Variable used in the name of thresholds and predictio files.
    '''
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))
    cont_preds = np.zeros((num_images, 3))
    cont_labels = np.zeros((num_images, 3))

    with torch.no_grad():
        model.to(device)
        model.eval()
        indx = 0
        print ('starting testing')
        for batch in iter(loader):
            labels_cat = batch[-2]
            labels_cont = batch[-1]
            imgs = []
            for img in batch[:-2]:
                imgs.append(img.to(device))

            pred_cat, pred_cont = model(*imgs)

            cat_preds[ indx : (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[ indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            cont_preds[ indx : (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
            cont_labels[ indx : (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
            indx = indx + pred_cat.shape[0]

    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    cont_preds = cont_preds.transpose()
    cont_labels = cont_labels.transpose()
    print ('completed testing')

    # Mat files used for emotic testing (matlab script)
    scipy.io.savemat(os.path.join(result_dir, '%s_cat_preds.mat' %(test_type)), mdict={'cat_preds':cat_preds})
    scipy.io.savemat(os.path.join(result_dir, '%s_cat_labels.mat' %(test_type)), mdict={'cat_labels':cat_labels})
    scipy.io.savemat(os.path.join(result_dir, '%s_cont_preds.mat' %(test_type)), mdict={'cont_preds':cont_preds})
    scipy.io.savemat(os.path.join(result_dir, '%s_cont_labels.mat' %(test_type)), mdict={'cont_labels':cont_labels})
    print ('saved mat files')

    test_scikit_ap(cat_preds, cat_labels, ind2cat)
    test_vad(cont_preds, cont_labels, ind2vad)
    thresholds = get_thresholds(cat_preds, cat_labels)
    np.save(os.path.join(result_dir, '%s_thresholds.npy' %(test_type)), thresholds)
    print ('saved thresholds')
    # calc_confusion(cat_preds, cat_labels, ind2cat, thresholds, save_path=result_dir)


def test_emotic(result_path, model_path, model_pretrained, context_norm, body_norm, face_norm, args):
    ''' Prepare test data and test models on the same.
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training. 
    :param context_norm: List containing mean and std values for context images.
    :param body_norm: List containing mean and std values for body images. 
    :param args: Runtime arguments.
    '''
    logger = logging.getLogger('Experiment')
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
    #Load data preprocessed npy files
    if args.context_mask:
        test_context = np.load(os.path.join(args.data_path, 'test_mask_context_arr.npy'))
    else:
        test_context = np.load(os.path.join(args.data_path, 'test_context_arr.npy'))
    test_body = np.load(os.path.join(args.data_path, 'test_body_arr.npy'))
    test_cat = np.load(os.path.join(args.data_path, 'test_cat_arr.npy'))
    test_cont = np.load(os.path.join(args.data_path, 'test_cont_arr.npy'))
    test_face = np.stack((np.load(os.path.join(args.data_path, 'test_face_arr.npy')),) * 3, axis=-1)
    logger.info('test  context ' + str(test_context.shape) + ' body ' + str(test_body.shape) + ' face ' +
                str(test_face.shape) + ' cat ' + str(test_cat.shape) + ' cont ' + str(test_cont.shape))

    # Initialize Dataset and Model
    test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    if args.body_model == 'swin_t':
        test_transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(size=[232],
                                                               interpolation=transforms.InterpolationMode.BICUBIC),
                                             transforms.CenterCrop(size=[224]),
                                             transforms.ToTensor()])

    face_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    if args.arch == 'double_stream':
        test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, test_transform, context_norm, body_norm)
        emotic_model = prep_models_double_stream(context_model=args.context_model, body_model=args.body_model,
                                                 args=args)
    else:
        test_dataset = EmoticTripleStreamDataset(test_context, test_body, test_face, test_cat, test_cont,
                                                test_transform, test_transform, face_test_transform,
                                                context_norm, body_norm, face_norm)
        if args.arch == 'triple_stream':
            emotic_model = prep_models_triple_stream(context_model=args.context_model, body_model=args.body_model,
                                                     face_model=args.face_model, args=args)
        elif args.arch == 'single_face':
            emotic_model = prep_models_single_face(face_model=args.face_model, args=args)
        elif args.arch == 'quadruple_stream':
            emotic_model = prep_models_quadruple_stream(context_model=args.context_model, body_model=args.body_model,
                                                face_model=args.face_model, caption_model=args.caption_model, args=args)
        else:
            emotic_model = prep_models_multistream(context_model=args.context_model, body_model=args.body_model,
                                                face_model=args.face_model, caption_model=args.caption_model, args=args)

    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    logger.info('test loader {}'.format(len(test_loader)))

    # resume of trained models
    if model_pretrained:
        ckpt = torch.load(model_pretrained)
    else:
        ckpt = torch.load(os.path.join(model_path, 'best_checkpoint.pth'))
    emotic_model.load_state_dict(ckpt['state_dict'])
    logger.info('Succesfully loaded models:')
    logger.info('model brief: {}'.format(ckpt['brief']))
    logger.info('model best mAP: {}'.format(ckpt['best_mAP']))

    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    test_data(emotic_model, device, test_loader, ind2cat, ind2vad, len(test_dataset), result_dir=result_path, test_type='test')


def test_data_caer(model, device, loader, ind2cat, num_images, result_dir='./', test_type='val'):
    ''' Test models on data
    :param models: List containing model_context, model_body and emotic_model (fusion model) in that order.
    :param device: Torch device. Used to send tensors to GPU if available.
    :param loader: Dataloader iterating over dataset.
    :param ind2cat: Dictionary converting integer index to categorical emotion.
    :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance)
    :param num_images: Number of images in the dataset.
    :param result_dir: Directory path to save results (predictions mat object and thresholds npy object).
    :param test_type: Test type variable. Variable used in the name of thresholds and predictio files.
    '''
    logger = logging.getLogger('Experiment')
    cat_preds = np.zeros(num_images)
    cat_labels = np.zeros(num_images)
    ind2cat = [ind2cat[i] for i in range(7)]
    cor = 0

    with torch.no_grad():
        model.to(device)
        model.eval()
        indx = 0
        print ('starting testing')
        for batch in iter(loader):
            labels_cat = batch[-1].to(device).long()
            imgs = []
            for img in batch[:-1]:
                imgs.append(img.to(device))
            pred_cat, pred_cont = model(*imgs)
            pred = pred_cat.argmax(dim=1)
            cor += torch.sum(pred == labels_cat)
            # for i in range(indx, indx + pred_cat.shape[0]):
            #     cat_preds[i, pred[i - indx]] = 1
            #     cat_labels[i, labels_cat[i - indx]] = 1
            cat_preds[indx: (indx + pred_cat.shape[0])] = pred.to("cpu").data.numpy()
            cat_labels[indx: (indx + labels_cat.shape[0])] = labels_cat.to("cpu").data.numpy()
            indx = indx + pred_cat.shape[0]

    # cat_preds = cat_preds.transpose()
    # cat_labels = cat_labels.transpose()
    cat_lens = []
    for i in range(7):
        filter = cat_labels == i
        cat_len = np.sum(filter)
        cat_lens.append(cat_len)
        cat_acc = np.sum(cat_preds[filter] == i) / cat_len
        logger.info('{}: {:.6f}'.format(ind2cat[i], cat_acc))
    logger.info('completed testing, Acc: {}'.format(cor / num_images))

    cm = confusion_matrix(cat_preds, cat_labels).astype(float)
    for i in range(7):
        cm[i] = cm[i] / cat_lens[i]
    ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=ind2cat, yticklabels=ind2cat)
    # ax.set_xticklabels(ind2cat, rotation=45, ha='right')
    # ax.set_yticklabels(ind2cat)
    plt.xlabel('Pred')
    plt.ylabel('GT')
    plt.savefig(os.path.join(result_dir, 'CM.png'))


def test_caer(result_path, model_path, model_pretrained, context_norm, body_norm, face_norm, args):
    ''' Prepare test data and test models on the same.
    :param result_path: Directory path to save the results (val_predidictions mat object, val_thresholds npy object).
    :param model_path: Directory path to load pretrained base models and save the models after training.
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
    #Load data preprocessed npy files
    # Initialize Dataset and Model
    test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    face_test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    if args.arch == 'caer_multistream':
        test_dataset = CAERDataset(args.data_path, os.path.join(args.data_path, 'test.txt'), test_transform,
                                   face_test_transform, context_norm, face_norm)
        caer_model = prep_models_caer_multistream(context_model=args.context_model, face_model=args.face_model,
                                                  caption_model=args.caption_model, args=args)
    else:
        raise ValueError('No such arch for caer')

    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    logger.info('test loader {}'.format(len(test_loader)))

    # resume of trained models
    if model_pretrained:
        ckpt = torch.load(model_pretrained)
    else:
        ckpt = torch.load(os.path.join(model_path, 'best_checkpoint.pth'))
    caer_model.load_state_dict(ckpt['state_dict'])
    logger.info('Succesfully loaded models:')
    logger.info('model brief: {}'.format(ckpt['brief']))
    logger.info('model best Acc: {}'.format(ckpt['best_Acc']))

    device = torch.device("cuda:%s" %(str(args.gpu)) if torch.cuda.is_available() else "cpu")
    test_data_caer(caer_model, device, test_loader, ind2cat, len(test_dataset), result_dir=result_path, test_type='test')
