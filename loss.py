import torch
import torch.nn as nn
import torch.nn.functional as F
class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.bceloss_fn = nn.BCEWithLogitsLoss()
    def forward(self,preds, targets, smooth=1e-6, bce=True):
        dice_loss=0
        if bce:
            BCE =self.bceloss_fn(preds, targets)
            dice_loss+=BCE
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()
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