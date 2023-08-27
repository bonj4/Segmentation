import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 20
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_FILE = "checkpoint.pth.tar"
LEARNING_RATE = 3e-4
BATCH_SIZE = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 780  # 1280 originally
IMAGE_WIDTH = 1280  # 1918 originally
TRAIN_IMG_DIR = "/home/visio-ai/PycharmProjects/FaceAndPlate/Dataset/seg/JPEGImages"
TRAIN_MASK_DIR = "/home/visio-ai/PycharmProjects/FaceAndPlate/Dataset/seg/SegmentationClass"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

train_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        #     max_pixel_value=255.0,
        # ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # A.Normalize(
        #     mean=[0.0, 0.0, 0.0],
        #     std=[1.0, 1.0, 1.0],
        #     max_pixel_value=255.0,
        # ),
        ToTensorV2(),
    ],
)