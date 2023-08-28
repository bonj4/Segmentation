import cv2
from unet import UNET
from utils import load_checkpoint
from  config import *

def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("checkpoint40.pth.tar"), model)

    # train_loader ,val_loader= get_loaders(
    #     TRAIN_IMG_DIR,
    #     TRAIN_MASK_DIR,
    #     VAL_IMG_DIR,
    #     VAL_MASK_DIR,
    #     BATCH_SIZE,
    #     train_transform,
    #     val_transforms,
    #     NUM_WORKERS,
    #     PIN_MEMORY,
    # )
    #
    #
    # save_predictions_as_imgs(
    #     val_loader, model, folder="saved_images/", device=DEVICE
    # )
    path=r""
    image=cv2.imread(path)
    image=cv2.resize(image,(image.shape[1]//8,image.shape[0]//8))
    x=torch.from_numpy(image)
    x = x.to(device='cuda').unsqueeze(0).permute(0,3,1,2).float()
    with torch.no_grad():
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
        preds=preds.squeeze(0).squeeze(0).cpu().numpy()
        preds*=255 # convert to 0-255 scale for cv2.imshow function to work properly
        cv2.imshow("winname",preds)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main()