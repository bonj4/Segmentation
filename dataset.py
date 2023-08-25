import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from config import *
class Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"),dtype=np.float32)

        try:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
            mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)

        except:
            mask = np.zeros(image.shape[:2])
        mask/= 255.0
        image/= 255.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
# class Dataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None):
#
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.transform = transform
#         self.annotations = os.listdir(image_dir)
#
#     def __len__(self):
#         return len(self.annotations)
#
#     def __getitem__(self, index):
#         img_path = os.path.join(self.image_dir, self.annotations[index])
#         image = np.array(cv2.imread(img_path))
#         label_path = os.path.join(self.label_dir, self.annotations[index][:-5] + ".txt")
#         mask =  self._convert_polygon_to_mask(file_path=label_path,image=image)
#         # mask[mask == 255] = 1.0
#
#         if self.transform is not None:
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]
#
#         return image,mask
#     def _convert_polygon_to_mask(self,file_path,image):
#         mask = np.zeros(image.shape[:2], np.uint8)
#         # Step 1: Read the text file line by line
#         with open(file_path, 'r') as file:
#             for line in file:
#                 array = []
#                 elements = line.strip().split()[1]
#                 elements=elements.split(",")
#                 for idx in range(0, len(elements), 2):
#                     array.append([int(float(elements[idx])), int(float(elements[idx + 1]))])
#                 # print(array)
#                 array=np.array(array)
#                 mask = cv2.drawContours(mask, [array], -1, (255), -1, cv2.LINE_AA)
#         return mask

def test():

    transform = val_transforms
    dataset = Dataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        transform=transform,
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        x=x.squeeze(0).permute(1,2,0).numpy()
        y=y.squeeze(0).numpy()
        cv2.namedWindow("x", cv2.WINDOW_NORMAL)
        cv2.imshow("x", x)
        cv2.namedWindow("y", cv2.WINDOW_NORMAL)
        cv2.imshow("y", y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break


if __name__ == "__main__":
    test()