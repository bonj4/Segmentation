import torch
import torch.nn as nn
import torch.nn.functional as F
from piqa import SSIM
from torch.autograd import Variable


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True,bce=True):
        super(DiceLoss, self).__init__()
        self.bceloss_fn = nn.BCEWithLogitsLoss()
        self.bce=bce
    def forward(self,preds, targets, smooth=1e-6):
        assert preds.shape[0] == targets.shape[0], "predict & target batch size don't match"
        dice_loss=0
        if self.bce:
            BCE =self.bceloss_fn(preds, targets)
            dice_loss+=BCE
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = torch.sigmoid(preds)
        # preds = (preds > 0.5).float()
        #flatten label and prediction tensors
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (preds * targets).sum()
        dice_loss +=1- (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)

        return dice_loss

def to_one_hot(tensor, nClasses):
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, nClasses, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


#https://discuss.pytorch.org/t/how-to-implement-soft-iou-loss/15152
class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union
        ## Return average loss over classes and batch
        return loss.mean()
# IMPLEMENTATION CREDIT: https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)