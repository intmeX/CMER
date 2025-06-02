import os
import torch
import torch.nn as nn
import logging
import torchvision.models as models
from torchsummary import summary
from model.fer_cnn import load_trained_sfer
from model.clip import ClipCaptain


class SKConv1D(nn.Module):
    
    def __init__(
            self,
            features,
            M,
            G,
            r,
            stride=1,
            L=64,
    ):
        super(SKConv1D, self).__init__()
        self.L = L
        self.num_channel = int(features / L)
        self.d = int(self.num_channel / r)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv1d(self.num_channel, self.num_channel, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                # nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm1d(self.num_channel),
                # nn.BatchNorm2d(self.num_channel),
                nn.ReLU()
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.num_channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(self.d, self.num_channel)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, self.num_channel, self.L)
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze(-1)
        # fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1).view(-1, self.features)
        return self.softmax(fea_v)


class SESeg1D(nn.Module):

    def __init__(
            self,
            features,
            r,
            L=64,
    ):
        super(SESeg1D, self).__init__()
        self.L = L
        self.num_channel = int(features / L)
        self.d = int(self.num_channel / r)
        self.features = features
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_channel, self.d),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d, self.num_channel),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.view(-1, self.num_channel, self.L)
        fea_s = self.gap(x).squeeze(-1)
        fea_z = self.fc1(fea_s)
        attn = self.fc2(fea_z).view(-1, self.num_channel, 1)
        x = x * attn.expand_as(x)
        return x.view(-1, self.features)


class Emotic(nn.Module):
    ''' Emotic Model'''

    def __init__(self, num_context_features, num_body_features, model_context, model_body):
        super(Emotic, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.model_context = model_context
        self.model_body = model_body
        self.fuse = nn.Sequential(
            nn.Linear((self.num_context_features + self.num_body_features), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        # self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.d1 = nn.Dropout(p=0.5)
        # self.relu = nn.ReLU()
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'DoubleStreamNet'

    def forward(self, x_context, x_body):
        context_features = self.model_context(x_context).view(-1, self.num_context_features)
        body_features = self.model_body(x_body).view(-1, self.num_body_features)
        fuse_features = torch.cat((context_features, body_features), 1)
        # fuse_out = self.fc1(fuse_features)
        # fuse_out = self.bn1(fuse_out)
        # fuse_out = self.relu(fuse_out)
        fuse_out = self.fuse(fuse_features)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class EmoticTripleStream(nn.Module):
    ''' Triple Stream Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_face_features, model_context, model_body, model_face):
        super(EmoticTripleStream, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.model_context = model_context
        self.model_body = model_body
        self.model_face = model_face
        self.fuse = nn.Sequential(
            nn.Linear((self.num_context_features + self.num_body_features + self.num_face_features), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'TripleStreamNet'

    def forward(self, x_context, x_body, x_face):
        context_features = self.model_context(x_context).view(-1, self.num_context_features)
        body_features = self.model_body(x_body).view(-1, self.num_body_features)
        face_features = self.model_face(x_face).view(-1, self.num_face_features)
        fuse_features = torch.cat((context_features, body_features, face_features), 1)
        fuse_out = self.fuse(fuse_features)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class EmoticQuadrupleStream(nn.Module):
    ''' Triple Stream Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_face_features, num_caption_feature, model_context,
                 model_body, model_face, model_caption):
        super(EmoticQuadrupleStream, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.num_caption_features = num_caption_feature
        self.model_context = model_context
        self.model_body = model_body
        self.model_face = model_face
        self.model_caption = model_caption
        self.fuse = nn.Sequential(
            nn.Linear((self.num_context_features + self.num_body_features + self.num_face_features + self.num_caption_features), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'QuadrupleStreamNet'

    def forward(self, x_context, x_body, x_face):
        context_features = self.model_context(x_context).view(-1, self.num_context_features)
        body_features = self.model_body(x_body).view(-1, self.num_body_features)
        face_features = self.model_face(x_face).view(-1, self.num_face_features)
        caption_features = self.model_caption(x_context).view(-1, self.num_caption_features)
        fuse_features = torch.cat((context_features, body_features, face_features, caption_features), 1)
        fuse_out = self.fuse(fuse_features)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class EmoticSingleFace(nn.Module):
    ''' Face Stream Emotic Model'''

    def __init__(self, num_face_features, model_face):
        super(EmoticSingleFace, self).__init__()
        self.num_face_features = num_face_features
        self.model_face = model_face
        self.fc_cat = nn.Linear(self.num_face_features, 26)
        self.fc_cont = nn.Linear(self.num_face_features, 3)
        self.brief = 'SingleFace'

    def forward(self, x_context, x_body, x_face):
        face_features = self.model_face(x_face).view(-1, self.num_face_features)
        cat_out = self.fc_cat(face_features)
        cont_out = self.fc_cont(face_features)
        return cat_out, cont_out


class EmoticSingle(nn.Module):
    ''' Face Stream Emotic Model'''

    def __init__(self, num_features, model, arch):
        super(EmoticSingle, self).__init__()
        self.num_features = num_features
        self.model = model
        self.fc_cat = nn.Linear(self.num_face_features, 26)
        self.fc_cont = nn.Linear(self.num_face_features, 3)
        self.brief = arch

    def forward(self, x_context, x_body, x_face):
        if self.brief == 'SingleFace':
            features = self.model(x_face).view(-1, self.num_features)
        elif self.brief == 'SingleBody':
            features = self.model(x_body).view(-1, self.num_features)
        elif self.brief == 'SingleContext':
            features = self.model(x_context).view(-1, self.num_features)
        else:
            features = self.model(x_context).view(-1, self.num_features)
        cat_out = self.fc_cat(features)
        cont_out = self.fc_cont(features)
        return cat_out, cont_out


class EmoticMultiStream(nn.Module):
    ''' Multi-Stream Emotic Model'''

    def __init__(self, num_features, models, stream_bit, fusion, fuse_l, fuse_r):
        super(EmoticMultiStream, self).__init__()
        self.num_features = num_features
        self.models = nn.ModuleList(models)
        self.stream_bit = stream_bit
        self.fuse_dim = 0
        for i in range(len(stream_bit)):
            if self.stream_bit[i]:
                self.fuse_dim += self.num_features[i]
        if fusion == 'se_fusion':
            self.attn = SESeg1D(self.fuse_dim, fuse_r, fuse_l)
        else:
            self.attn = nn.Sequential()
        self.fuse = nn.Sequential(
            nn.Linear(self.fuse_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'MultiStream{}{}{}{}'.format(*self.stream_bit)

    def forward(self, x_context, x_body, x_face):
        xs = [x_context, x_body, x_face, x_context]
        features = []
        for i in range(len(xs)):
            if self.stream_bit[i]:
                features.append(self.models[i](xs[i]).view(-1, self.num_features[i]))
        features = self.attn(torch.cat(features, 1))
        features = self.fuse(features)
        cat_out = self.fc_cat(features)
        cont_out = self.fc_cont(features)
        return cat_out, cont_out


class SKEmoticQuadrupleStream(nn.Module):
    ''' Triple Stream Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_face_features, num_caption_feature, model_context,
                 model_body, model_face, model_caption):
        super(SKEmoticQuadrupleStream, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.num_caption_features = num_caption_feature
        self.model_context = model_context
        self.model_body = model_body
        self.model_face = model_face
        self.model_caption = model_caption
        self.context_attn = SKConv1D(self.num_context_features + self.num_caption_features, 2, 1, 4)
        self.body_attn = SKConv1D(self.num_body_features + self.num_face_features, 2, 1, 4)
        self.fuse = nn.Sequential(
            nn.Linear((self.num_context_features + self.num_body_features + self.num_face_features + self.num_caption_features), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'SKQuadrupleStreamNet'

    def forward(self, x_context, x_body, x_face):
        context_features = self.model_context(x_context).view(-1, self.num_context_features)
        body_features = self.model_body(x_body).view(-1, self.num_body_features)
        face_features = self.model_face(x_face).view(-1, self.num_face_features)
        caption_features = self.model_caption(x_context).view(-1, self.num_caption_features)
        context_fuse_feature = torch.cat([context_features, caption_features], 1)
        context_fuse_feature = self.context_attn(context_fuse_feature) * context_fuse_feature
        body_fuse_feature = torch.cat([body_features, face_features], 1)
        body_fuse_feature = self.body_attn(body_fuse_feature) * body_fuse_feature
        # fuse_features = torch.cat((context_features, body_features, face_features, caption_features), 1)
        fuse_features = torch.cat((context_fuse_feature, body_fuse_feature), 1)
        fuse_out = self.fuse(fuse_features)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class SEEmoticQuadrupleStream(nn.Module):
    ''' Quad Stream Emotic Model'''

    def __init__(self, num_context_features, num_body_features, num_face_features, num_caption_feature, model_context,
                 model_body, model_face, model_caption, r, L, fuse_2_layer=False):
        super(SEEmoticQuadrupleStream, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.num_caption_features = num_caption_feature
        self.model_context = model_context
        self.model_body = model_body
        self.model_face = model_face
        self.model_caption = model_caption
        self.context_fuse = SESeg1D(self.num_context_features + self.num_caption_features, r=r, L=L)
        self.body_fuse = SESeg1D(self.num_body_features + self.num_face_features, r=r, L=L)
        self.fuse_2_layer = fuse_2_layer
        self.fuse_len = self.num_context_features + self.num_body_features + self.num_face_features + self.num_caption_features
        if not self.fuse_2_layer:
            self.fuse_2 = nn.Sequential()
        else:
            self.fuse_2 = SESeg1D(self.fuse_len, r=r, L=L)
        self.fuse = nn.Sequential(
            nn.Linear(self.fuse_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'SEQuadrupleStreamNet'

    def forward(self, x_context, x_body, x_face):
        context_features = self.model_context(x_context).view(-1, self.num_context_features)
        body_features = self.model_body(x_body).view(-1, self.num_body_features)
        face_features = self.model_face(x_face).view(-1, self.num_face_features)
        caption_features = self.model_caption(x_context).view(-1, self.num_caption_features)
        context_fuse_feature = torch.cat([context_features, caption_features], 1)
        context_fuse_feature = self.context_fuse(context_fuse_feature)
        body_fuse_feature = torch.cat([body_features, face_features], 1)
        body_fuse_feature = self.body_fuse(body_fuse_feature)
        fuse_features = torch.cat((context_fuse_feature, body_fuse_feature), 1)
        fuse_features = self.fuse_2(fuse_features)
        fuse_out = self.fuse(fuse_features)
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out


class CaerMultiStream(nn.Module):
    ''' Multi-Stream CAER Model'''

    def __init__(self, num_features, models, fusion, fuse_l, fuse_r):
        super(CaerMultiStream, self).__init__()
        self.num_features = num_features
        self.models = nn.ModuleList(models)
        self.fuse_dim = 0
        for i in range(len(num_features)):
            self.fuse_dim += self.num_features[i]
        self.first_fuse_dim = self.num_features[0] + self.num_features[1]
        if fusion == 'se_fusion':
            self.first_attn = SESeg1D(self.first_fuse_dim, fuse_r, fuse_l)
            self.attn = SESeg1D(self.fuse_dim, fuse_r, fuse_l)
        else:
            self.first_attn = nn.Sequential()
            self.attn = nn.Sequential()
        self.fuse = nn.Sequential(
            nn.Linear(self.fuse_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc_cat = nn.Linear(256, 7)
        self.fc_cont = nn.Linear(256, 3)
        self.brief = 'CaerMultiStream'

    def forward(self, x_context, x_face):
        xs = [x_context, x_face, x_context]
        features = []
        for i in range(len(xs)):
            features.append(self.models[i](xs[i]).view(-1, self.num_features[i]))
        features_low = self.first_attn(torch.cat(features[: 2], 1))
        features = self.attn(torch.cat([features[2], features_low], 1))
        features = self.fuse(features)
        cat_out = self.fc_cat(features)
        cont_out = self.fc_cont(features)
        return cat_out, cont_out


def prep_models(context_model='resnet50', body_model='resnet50', model_dir='./'):
    ''' Download imagenet pretrained models for context_model and body_model.
    :param context_model: Model to use for conetxt features.
    :param body_model: Model to use for body features.
    :param model_dir: Directory path where to store pretrained models.
    :return: Yolo model after loading model weights
    '''
    model_name = '%s_places365.pth.tar' % context_model
    model_file = os.path.join(model_dir, model_name)
    if not os.path.exists(model_file):
        download_command = 'wget ' + 'http://places2.csail.mit.edu/models_places365/' + model_name + ' -O ' + model_file
        os.system(download_command)

    save_file = os.path.join(model_dir, '%s_places365_py36.pth.tar' % context_model)
    # from functools import partial
    # import pickle
    # pickle.load = partial(pickle.load, encoding="latin1")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
    model = torch.load(model_file, map_location=lambda storage, loc: storage)
    torch.save(model, save_file)

    # create the network architecture
    model_context = models.__dict__[context_model](num_classes=365)
    checkpoint = torch.load(save_file, map_location=lambda storage,
                                                           loc: storage)  # model trained in GPU could be deployed in CPU machine like this!
    if context_model == 'densenet161':
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        state_dict = {str.replace(k, 'norm.', 'norm'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'conv.', 'conv'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normweight', 'norm.weight'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normrunning', 'norm.running'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'normbias', 'norm.bias'): v for k, v in state_dict.items()}
        state_dict = {str.replace(k, 'convweight', 'conv.weight'): v for k, v in state_dict.items()}
    else:
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
            'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
    model_context.load_state_dict(state_dict)
    model_context.eval()
    model_context.cpu()
    torch.save(model_context, os.path.join(model_dir, 'context_model' + '.pth'))

    print('completed preparing context model')

    model_body = models.__dict__[body_model](pretrained=True)
    model_body.cpu()
    torch.save(model_body, os.path.join(model_dir, 'body_model' + '.pth'))

    print('completed preparing body model')
    emotic_model = Emotic(model_context, model_body)
    return emotic_model


def build_context_model(context_model, args):
    logger = logging.getLogger('Experiment')
    model_context = models.__dict__[context_model](weights='DEFAULT')
    num_context_features = list(model_context.children())[-1].in_features
    model_context = nn.Sequential(*(list(model_context.children())[:-1]))
    logger.info('completed preparing context model: {}'.format(context_model))
    if args.context_model_frozen:
        for param in model_context.parameters():
            param.requires_grad = False
        logger.info('context model frozen')
    logger.info(summary(model_context, (3, 224, 224), device="cpu"))
    logger.info('num of features: {}'.format(num_context_features))
    return num_context_features, model_context


def build_body_model(body_model, args):
    logger = logging.getLogger('Experiment')
    model_body = models.__dict__[body_model](weights='DEFAULT')
    num_body_features = list(model_body.children())[-1].in_features
    model_body = nn.Sequential(*(list(model_body.children())[:-1]))
    logger.info('completed preparing body model: {}'.format(body_model))
    if args.body_model_frozen:
        for param in model_body.parameters():
            param.requires_grad = False
        logger.info('body model frozen')
    logger.info(summary(model_body, (3, 128, 128), device="cpu"))
    logger.info('num of features: {}'.format(num_body_features))
    return num_body_features, model_body


def build_face_model(face_model, args):
    logger = logging.getLogger('Experiment')
    if face_model == 'sfer':
        model_face = load_trained_sfer()
        num_face_features = list(model_face.children())[-1].out_features
        '''
        elif face_model == 'ResEmoteNet':
            if args.face_weight == 'initial':
                model_face = ResEmoteNet()
                model_face.weight_init()
            else:
                model_face = load_res_emote()
            num_face_features = 256
        '''
    else:
        model_face = models.__dict__[face_model](weights='DEFAULT')
        num_face_features = list(model_face.children())[-1].in_features
        model_face = nn.Sequential(*(list(model_face.children())[:-1]))
    logger.info('completed preparing face model: {}'.format(face_model))
    if args.face_model_frozen:
        for param in model_face.parameters():
            param.requires_grad = False
        logger.info('face model frozen')
    logger.info(summary(model_face, (3, 48, 48), device="cpu"))
    logger.info('num of features: {}'.format(num_face_features))
    return num_face_features, model_face


def build_caption_model(caption_model, args):
    logger = logging.getLogger('Experiment')
    model_caption = ClipCaptain()
    logger.info('completed preparing caption model: {}'.format(caption_model))
    for param in model_caption.parameters():
        param.requires_grad = False
    logger.info('caption model frozen')
    num_caption_feature = 512
    # logger.info(summary(model_caption, (3, 224, 224), device="cpu"))
    logger.info('num of features: {}'.format(num_caption_feature))
    return num_caption_feature, model_caption


def prep_models_double_stream(context_model, body_model, args):
    # create the network architecture
    num_context_features, model_context = build_context_model(context_model, args)
    num_body_features, model_body = build_body_model(body_model, args)
    emotic_model = Emotic(num_context_features, num_body_features, model_context, model_body)
    return emotic_model


def prep_models_triple_stream(context_model, body_model, face_model, args):
    # create the network architecture
    num_context_features, model_context = build_context_model(context_model, args)
    num_body_features, model_body = build_body_model(body_model, args)
    num_face_features, model_face = build_face_model(face_model, args)
    emotic_model = EmoticTripleStream(num_context_features, num_body_features, num_face_features, model_context, model_body, model_face)
    return emotic_model


def prep_models_quadruple_stream(context_model, body_model, face_model, caption_model, args):
    # create the network architecture
    num_context_features, model_context = build_context_model(context_model, args)
    num_body_features, model_body = build_body_model(body_model, args)
    num_face_features, model_face = build_face_model(face_model, args)
    num_caption_features, model_caption = build_caption_model(caption_model, args)
    if args.fuse_model == 'default':
        emotic_model = EmoticQuadrupleStream(num_context_features, num_body_features, num_face_features,
                                             num_caption_features, model_context, model_body, model_face, model_caption)
    elif args.fuse_model == 'sk_fusion':
        emotic_model = SKEmoticQuadrupleStream(num_context_features, num_body_features, num_face_features,
                                             num_caption_features, model_context, model_body, model_face, model_caption)
    else:
        emotic_model = SEEmoticQuadrupleStream(num_context_features, num_body_features, num_face_features,
                                             num_caption_features, model_context, model_body, model_face, model_caption,
                                               args.fuse_r, args.fuse_L, args.fuse_2_layer)

    return emotic_model


def prep_models_single_face(face_model, args):
    # create the network architecture
    num_face_features, model_face = build_face_model(face_model, args)
    emotic_model = EmoticSingleFace(num_face_features, model_face)
    return emotic_model


def prep_models_multistream(context_model, body_model, face_model, caption_model, args):
    # create the network architecture
    stream_bit = []
    for i in args.stream_bit:
        if i not in ['0', '1']:
            raise ValueError('stream cannot be resolve:{}'.format(args.stream_bit))
        stream_bit.append(int(i))
    assert len(stream_bit) == 4
    num_context_features, model_context = build_context_model(context_model, args)
    num_body_features, model_body = build_body_model(body_model, args)
    num_face_features, model_face = build_face_model(face_model, args)
    num_caption_features, model_caption = build_caption_model(caption_model, args)
    emotic_model = EmoticMultiStream(
        [num_context_features, num_body_features, num_face_features, num_caption_features],
        [model_context, model_body, model_face, model_caption],
        stream_bit,
        args.fuse_model,
        args.fuse_L,
        args.fuse_r,
    )
    return emotic_model


def prep_models_caer_multistream(context_model, face_model, caption_model, args):
    # create the network architecture
    num_context_features, model_context = build_context_model(context_model, args)
    num_face_features, model_face = build_face_model(face_model, args)
    num_caption_features, model_caption = build_caption_model(caption_model, args)
    caer_model = CaerMultiStream(
        [num_context_features, num_face_features, num_caption_features],
        [model_context, model_face, model_caption],
        args.fuse_model,
        args.fuse_L,
        args.fuse_r,
    )
    return caer_model


if __name__ == '__main__':
    model = models.__dict__['resnet50'](num_classes=365)
    c = list(model.children())[-2]
    print(c.in_features)
    print(list(model.children())[-1].in_features)


