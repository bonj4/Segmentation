import torch
import torch.nn as nn


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