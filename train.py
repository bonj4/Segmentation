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
)
from config import *
import cv2
import numpy as np

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
        #for which photos the model predicted wrong ?
        if loss.item()>1.0:
            image=data.squeeze(0).permute(1,2,0).cpu()*255
            image=np.array(image,dtype=np.uint8)

            pred=nn.functional.sigmoid(predictions).squeeze(0).squeeze(0).detach().cpu()*255
            pred=np.array(pred,dtype=np.uint8)

            targets=targets.squeeze(0).squeeze(0).detach().cpu()*255
            targets=np.array(targets,dtype=np.uint8)

            cv2.imshow("pred",pred)
            cv2.imshow("targets",targets)
            cv2.imshow("image",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint40.pth.tar"), model)


    train_loader ,val_loader= get_loaders(
        train_img_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        test_img_dir=VAL_IMG_DIR,
        test_mask_dir=VAL_MASK_DIR,
        batch_size=1,
        train_transform=train_transform,
        val_transform=val_transforms,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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