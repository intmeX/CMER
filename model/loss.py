import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1,26))/26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)
        loss = (((pred - target)**2) * self.weights)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1,26))
        weights[target_stats != 0 ] = 1.0/torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001
        return weights


class BCEDiscreteLoss(nn.Module):
    ''' Class to measure BCE loss between categorical emotion predictions and labels.'''
    def __init__(self, weight_type='dynamic', static_pos_weights=None, device=torch.device('cpu')):
        super(BCEDiscreteLoss, self).__init__()
        self.device = device
        self.weight_type = weight_type
        self.pos_weights = static_pos_weights.to(self.device)
        self.weight = (self.pos_weights + 1) / (2 * self.pos_weights)
        self.static_func = nn.BCEWithLogitsLoss(weight=self.weight, pos_weight=self.pos_weights, reduction='sum')
        # self.weights = self.weights.to(self.device)

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            pos_weights = self.prepare_dynamic_weights(target).to(self.device)
            weight = (pos_weights + 1) / (2 * pos_weights)
            criterion = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weights, reduction='sum')
        else:
            criterion = self.static_func
        loss = criterion(pred, target)
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        pos_num = target.sum(axis=0, dtype=torch.float32)
        pos_weight = (target.shape[0] - pos_num + 1.0) / (pos_num + 1.0)
        return pos_weight


class FocalDiscreteLoss(nn.Module):
    def __init__(self, weight_type='dynamic', static_pos_weights=None, hard_gamma=2.0, device=torch.device('cpu')):
        super(FocalDiscreteLoss, self).__init__()
        self.device = device
        self.weight_type = weight_type
        self.hard_gamma = hard_gamma
        self.pos_weights = static_pos_weights.to(self.device)
        self.weight = (self.pos_weights + 1) / (2 * self.pos_weights)
        self.static_func = nn.BCEWithLogitsLoss(weight=self.weight, pos_weight=self.pos_weights, reduction='none')

    def forward(self, pred, target):
        probs = torch.sigmoid(pred)
        coef = torch.abs(target - probs) ** self.hard_gamma
        if self.weight_type == 'dynamic':
            pos_weights = self.prepare_dynamic_weights(target).to(self.device)
            weight = (pos_weights + 1) / (2 * pos_weights)
            criterion = nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weights, reduction='none')
        else:
            criterion = self.static_func
        # log_probs = torch.where(pred >= 0,
        #                         F.softplus(pred, -1, 50),
        #                         pred - F.softplus(pred, 1, 50))
        # log_1_probs = torch.where(pred >= 0,
        #                           -pred + F.softplus(pred, -1, 50),
        #                           -F.softplus(pred, 1, 50))
        # loss = target * self.pos_weights * log_probs + (1. - target) * log_1_probs
        # loss = target * self.pos_weights * torch.log(probs) + (1. - target) * torch.log(1.0 - probs)
        # loss = -loss * self.weight
        # loss = criterion(pred, target) * coef
        loss = criterion(pred, target)
        loss = loss * coef
        return loss.sum()

    def prepare_dynamic_weights(self, target):
        pos_num = target.sum(axis=0, dtype=torch.float32)
        pos_weight = (target.shape[0] - pos_num + 1.0) / (pos_num + 1.0)
        return pos_weight


class ZLPRLoss(nn.Module):
    def __init__(self):
        super(ZLPRLoss, self).__init__()

    def forward(self, pred, target):
        output = (1 - 2 * target) * pred
        output_neg = output - target * 1e12
        output_pos = output - (1 - target) * 1e12
        zeros = torch.zeros_like(output[:, :1])
        output_neg = torch.cat([output_neg, zeros], dim=1)
        output_pos = torch.cat([output_pos, zeros], dim=1)
        loss_pos = torch.logsumexp(output_pos, dim=1)
        loss_neg = torch.logsumexp(output_neg, dim=1)
        loss = (loss_neg + loss_pos).sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # maintain the class dim
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=0)
        union = pred.sum(dim=0) + target.sum(dim=0)
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth))
        return dice_loss.sum()


class ContinuousLoss_L2(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using l2 loss as base. '''
    def __init__(self, margin=1):
        super(ContinuousLoss_L2, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = labs ** 2
        loss[ (labs < self.margin) ] = 0.0
        return loss.sum()


class ContinuousLoss_SL1(nn.Module):
    ''' Class to measure loss between continuous emotion dimension predictions and labels. Using smooth l1 loss as base. '''
    def __init__(self, margin=1):
        super(ContinuousLoss_SL1, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        labs = torch.abs(pred - target)
        loss = 0.5 * (labs ** 2)
        loss[ (labs > self.margin) ] = labs[ (labs > self.margin) ] - 0.5
        return loss.sum()


def _test_discrete():
    # Discrete Loss function test
    target = torch.zeros((2,26))
    target[0, 0:13] = 1
    target[1, 13:] = 2
    target[:, 13] = 0

    pred = torch.ones((2,26)) * 1
    target = target.cuda()
    pred = pred.cuda()
    pred.requires_grad = True
    target.requires_grad = False

    disc_loss = DiscreteLoss('dynamic', torch.device("cuda:0"))
    loss = disc_loss(pred, target)
    print ('discrete loss class', loss, loss.shape, loss.dtype, loss.requires_grad)  # loss = 37.1217

    #Continuous Loss function test
    target = torch.ones((2,3))
    target[0, :] = 0.9
    target[1, :] = 0.2
    target = target.cuda()
    pred = torch.ones((2,3))
    pred[0, :] = 0.7
    pred[1, :] = 0.25
    pred = pred.cuda()
    pred.requires_grad = True
    target.requires_grad = False

    cont_loss_SL1 = ContinuousLoss_SL1()
    loss = cont_loss_SL1(pred*10, target * 10)
    print ('continuous SL1 loss class', loss, loss.shape, loss.dtype, loss.requires_grad) # loss = 4.8750

    cont_loss_L2 = ContinuousLoss_L2()
    loss = cont_loss_L2(pred*10, target * 10)
    print ('continuous L2 loss class', loss, loss.shape, loss.dtype, loss.requires_grad) # loss = 12.0


if __name__ == '__main__':
    # _test_discrete()
    target = torch.zeros((2, 26))
    target[0, 0:13] = 1
    target[1, 13:] = 1
    target[:, 13] = 0

    # pred = torch.ones((2, 26)) * 1
    pred = target.clone()
    pred = (pred * 2 - 1) * 5
    pred[:, 13] = 5
    pos_weight = torch.ones(26)
    Bce = FocalDiscreteLoss(weight_type='static', static_pos_weights=pos_weight, hard_gamma=2.0)
    print(Bce(pred, target))
    Bce = BCEDiscreteLoss(weight_type='static', static_pos_weights=pos_weight)
    print(Bce(pred, target))
    # nBce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.ones(26))
    # print(nBce(pred, target))
    # Dice = DiceLoss()
    # print(Dice(pred, target))
    # Zlpr = ZLPRLoss()
    # print(Zlpr(pred, target))
