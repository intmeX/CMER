config = {
    'mode': 'train_test',
    'data_path': r'C:\koe\DataCenter\emotic_npy',
    'context_model': 'resnet50',
    'context_model_frozen': True,
    'body_model': 'swin_t',
    'arch': 'double_stream',
    'discrete_loss_weight_type': 'dynamic',
    'continuous_loss_type': 'L2',
    'epochs': 1,
    'batch_size': 32,
    'optimizer': 'Adam',
    'sgd_momentum': 0.5,
    'learning_rate': 0.001,
    'scheduler': 'exp',
    'warmup': 1000,
}
