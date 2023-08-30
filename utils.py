import torch
from torch.utils.data import  DataLoader
from dataset import Dataset
import torchvision
import torch.nn.functional as F
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            # y = y.to(device).permute(0,3,1,2)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def check_class_accuracy_for_multiclasses(n_class,loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    #background,face,plate
    for cls in range(1,n_class):
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device).detach().cpu().numpy()
                preds = F.softmax(model(x)).squeeze(0).detach().cpu().numpy()
                mask=np.all(preds == [cls], axis=0,).astype(float)
                # mask = (mask > 0.5).float()

                num_correct += (mask == y).sum()
                num_pixels += preds.shape[0]*preds.shape[1]*preds.shape[2]
                dice_score += (2 * (mask * y).sum()) / (
                        (mask + y).sum() + 1e-8
                )
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_img_dir,
    train_mask_dir,
    test_img_dir,
    test_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds =Dataset(image_dir=train_img_dir,
                      mask_dir=train_mask_dir,
                      transform=train_transform
                      )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    val_ds = Dataset(image_dir=train_img_dir,
                     mask_dir=train_mask_dir,
                     transform=val_transform
                     )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
    )


    return train_loader, val_loader


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()