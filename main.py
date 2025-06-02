import importlib
import argparse
import logging
import random
import shutil
import torch
import time
import os
import numpy as np
from learn.train import train_emotic, train_caer
from learn.test import test_emotic, test_caer


def parse_args():
    parser = argparse.ArgumentParser()
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument('--config', type=str, default='')
    args, _ = config_parser.parse_known_args()
    if args.config:
        spec = importlib.util.spec_from_file_location('config', args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        config_dict = config.config

    exp_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--mode', type=str, default='train_test', choices=['train', 'test', 'train_test', 'inference'])
    parser.add_argument('--trainer', type=str, default='emotic', choices=['emotic', 'caer'])
    parser.add_argument('--data_path', type=str, help='Path to preprocessed data npy files/ csv files')
    parser.add_argument('--experiment_root', type=str, default='./exp', help='root to save experiment files')
    parser.add_argument('--experiment_id', type=str, default='exp_{}_'.format(exp_time), help='id of save experiment files (results, models, logs)')
    parser.add_argument('--experiment_name', type=str, default='untitled', help='name of experiment')
    parser.add_argument('--model_dir_name', type=str, default='models', help='Name of the directory to save models')
    parser.add_argument('--result_dir_name', type=str, default='results', help='Name of the directory to save results(predictions, labels mat files)')
    parser.add_argument('--log_dir_name', type=str, default='logs', help='Name of the directory to save logs (train, val)')
    parser.add_argument('--model_pretrained', type=str, default='', help='path to weights of pretrained model')
    parser.add_argument('--inference_file', type=str, help='Text file containing image context paths and bounding box')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--context_model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'swin_t'], help='context model type')
    parser.add_argument('--context_model_frozen', action='store_true', help='context model param change in training')
    parser.add_argument('--context_mask', action='store_true', help='whether context image contains body')
    parser.add_argument('--body_model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'swin_t'], help='body model type')
    parser.add_argument('--body_model_frozen', action='store_true', help='body model param change in training')
    parser.add_argument('--face_model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'swin_t', 'sfer'], help='face model type')
    parser.add_argument('--face_weight', type=str, default='default', choices=['default', 'initial'], help='face model weights')
    parser.add_argument('--face_model_frozen', action='store_true', help='face model param change in training')
    parser.add_argument('--caption_model', type=str, default='CLIP', choices=['CLIP'], help='caption feature of context model type')
    parser.add_argument('--fuse_model', type=str, default='default', choices=['default', 'sk_fusion', 'se_fusion'], help='model for feature fusion')
    parser.add_argument('--fuse_L', type=int, default=64, help='hidden dim for fusion')
    parser.add_argument('--fuse_r', type=int, default=4, help='squeeze times for fusion')
    parser.add_argument('--fuse_2_layer', action='store_true', default=False, help='fusion layers')
    parser.add_argument('--stream_bit', type=str, default='1111', help='binary to show multistream backbone')
    parser.add_argument('--arch', type=str, default='double_stream', choices=[
        'double_stream', 'triple_stream', 'quadruple_stream', 'single_face', 'multi_stream', 'caer_multistream'
    ], help='overall architect')
    parser.add_argument('--pretrained', type=str, default='', help='the path of pretrained checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--scheduler', type=str, default='const', choices=['const', 'exp', 'cosine'])
    parser.add_argument('--decay_start', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.997)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--cat_loss_weight', type=float, default=0.5, help='weight for discrete loss')
    parser.add_argument('--cont_loss_weight', type=float, default=0.5, help='weight fot continuous loss')
    parser.add_argument('--continuous_loss_type', type=str, default='Smooth L1', choices=['L2', 'Smooth L1'], help='type of continuous loss')
    parser.add_argument('--discrete_loss_type', type=str, default='default', choices=['default', 'bce', 'dice', 'zlpr', 'focal', 'ce'], help='discrete loss')
    parser.add_argument('--hard_gamma', type=float, default=2.0)
    parser.add_argument('--discrete_loss_weight_type', type=str, default='dynamic', choices=['dynamic', 'mean', 'static'], help='weight policy for discrete loss')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)  # use batch size = double(categorical emotion classes)
    parser.add_argument('--num_worker', type=int, default=0)
    # override the default value
    for action in parser._actions:
        if action.dest in config_dict:
            action.default = config_dict[action.dest]
    # Generate args
    args, _ = parser.parse_known_args()
    print(args)
    return args


def check_paths(args):    
    ''' Check (create if they don't exist) experiment directories.
    :param args: Runtime arguments as passed by the user.
    :return: List containing result_dir_path, model_dir_path, train_log_dir_path, val_log_dir_path.
    '''
    folders = [args.result_dir_name, args.model_dir_name, args.log_dir_name]
    experiment_path = os.path.join(args.experiment_root, args.experiment_id + args.experiment_name)
    paths = list()
    for folder in folders:
        folder_path = os.path.join(experiment_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)

    '''
    log_folders = ['train']  # , 'val'
    for folder in log_folders:
        folder_path = os.path.join(args.experiment_path, args.log_dir_name, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths.append(folder_path)
    '''
    return paths


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_args()
    print ('mode ', args.mode)

    result_path, model_path, train_log_path = check_paths(args)

    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        seed_everything(args.seed)

    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    if args.body_model == 'swin_t':
        body_mean = [0.485, 0.456, 0.406]
        body_std = [0.229, 0.224, 0.225]
    face_mean = [0.507395516207, 0.507395516207, 0.507395516207]
    face_std = [0.255128989415, 0.255128989415, 0.255128989415]

    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]
    face_norm = [face_mean, face_std]

    logger = logging.getLogger('Experiment')
    logger.setLevel(logging.INFO)
    experiment_path = os.path.join(args.experiment_root, args.experiment_id + args.experiment_name)
    fh = logging.FileHandler(filename=os.path.join(experiment_path, 'experiment.log'), mode='a')
    fh.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(console_handler)

    if args.mode == 'train':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training')
        with open(os.path.join(experiment_path, f'config_{int(time.time())}.txt'), 'w') as f:
            print(args, file=f)
        if args.trainer == 'emotic':
            train_emotic(result_path, model_path, train_log_path, context_norm, body_norm, face_norm, args)
        else:
            train_caer(result_path, model_path, train_log_path, context_norm, body_norm, face_norm, args)
    elif args.mode == 'test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for testing')
        if args.trainer == 'emotic':
            test_emotic(result_path, model_path, args.model_pretrained, context_norm, body_norm, face_norm, args)
        else:
            test_caer(result_path, model_path, args.model_pretrained, context_norm, body_norm, face_norm, args)
    elif args.mode == 'train_test':
        if args.data_path is None:
            raise ValueError('Data path not provided. Please pass a valid data path for training and testing')
        with open(os.path.join(experiment_path, f'config_{int(time.time())}.txt'), 'w') as f:
            print(args, file=f)
        if args.trainer == 'emotic':
            train_emotic(result_path, model_path, train_log_path, context_norm, body_norm, face_norm, args)
        else:
            train_caer(result_path, model_path, train_log_path, context_norm, body_norm, face_norm, args)
        if args.trainer == 'emotic':
            test_emotic(result_path, model_path, args.model_pretrained, context_norm, body_norm, face_norm, args)
        else:
            test_caer(result_path, model_path, args.model_pretrained, context_norm, body_norm, face_norm, args)
    elif args.mode == 'inference':
        if args.inference_file is None:
            raise ValueError('Inference file not provided. Please pass a valid inference file for inference')
        raise NotImplementedError('No Inference scheme')
        # inference_emotic(args.inference_file, model_path, result_path, context_norm, body_norm, ind2cat, ind2vad, args)
    else:
        raise ValueError('Unknown mode')
