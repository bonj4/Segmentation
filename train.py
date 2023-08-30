import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from unet import UNET
from segnet import SegNet
from loss import DiceLoss,SSIMLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)
from  config import *
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # targets = targets.float().permute(0,3,1,2).to(device=DEVICE)


        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss(bce=True).to(DEVICE)
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
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        if epoch % 5==0:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if SAVE_MODEL:
                save_checkpoint(checkpoint)
            # check accuracy
            check_accuracy(val_loader, model, device=DEVICE)





if __name__ == "__main__":
    main()