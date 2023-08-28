import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
from loss import DiceLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
from  config import *

def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint40.pth.tar"), model)


    train_loader ,val_loader= get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    # check_accuracy(val_loader, model, device=DEVICE)


    save_predictions_as_imgs(
        val_loader, model, folder="saved_images/", device=DEVICE
    )





if __name__ == "__main__":
    main()