import os
import glob

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import RandomRotation, GaussianBlur, ToTensor, ColorJitter

class PlateDataset(Dataset):
    # Allowed characters on Russian license plates:
    CHARS = "0123456789ABEKMHOPCTYX"

    num_chars = len(CHARS) + 1

    # Label mappings (CTC requires blank = 0, so shift characters by +1)
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {i + 1: char for i, char in enumerate(CHARS)}
    BLANK_LABEL = 0  # for CTC loss

    def __init__(self, root_dir, mode="train", img_height=32, img_width=100):
        self.root_dir = root_dir
        self.mode = mode
        self.img_height = img_height
        self.img_width = img_width

        # Collect all image paths
        if mode == "train":
            simple_ = glob.glob(os.path.join(root_dir, "train", "simple", "*.png"))
            complex_ = glob.glob(os.path.join(root_dir, "train", "complex", "*.png"))
            self.paths = simple_ + complex_

        elif mode == "valid":
            simple_ = glob.glob(os.path.join(root_dir, "valid", "simple", "*.png"))
            complex_ = glob.glob(os.path.join(root_dir, "valid", "complex", "*.png"))
            self.paths = simple_ + complex_

        elif mode == "test_val":
            self.paths = glob.glob(os.path.join(root_dir, "test_val", "*.png"))

        elif mode == "test":
            self.paths = glob.glob(os.path.join(root_dir, "test", "*.png"))

        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def _extract_label(self, filename):
        basename = os.path.basename(filename)
        parts = basename.split("_")
        label_str = parts[2].split(".")[0] 
        return label_str


    def make_transformations(self, image):

        if self.mode == 'train':
            strong_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=15, expand=True),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                # transforms.RandomHorizontalFlip(p=0.2),
                transforms.ColorJitter(
                    brightness=0.3, 
                    contrast=0.3,
                    saturation=0.0, 
                    hue=0.0
            ),
                transforms.GaussianBlur(kernel_size=(3,7), sigma=(0.1, 2.0)),
                transforms.Resize((32, 160)),
                transforms.ToTensor(),
        ])
        else:
            strong_transforms = transforms.Compose([
                transforms.Resize((32, 160)),
                transforms.ToTensor(),
        ])

        return strong_transforms(image)

    def __getitem__(self, index):
        path = self.paths[index]

        # Load RGB and convert (model expects RGB)
        image = Image.open(path).convert("L")
        image = image.resize((self.img_width, self.img_height), Image.BILINEAR)

        image = self.make_transformations(image)

        if self.mode == 'test':
            return image, path
        
        # Extract text label
        label_str = self._extract_label(path)

        # Convert characters to numeric labels
        target = []
        for c in label_str:
            target.append(self.CHAR2LABEL[c])

        target = torch.LongTensor(target)
        target_length = torch.LongTensor([len(target)])

        return image, target, target_length


def plate_collate_fn(batch):

    if len(batch[0]) == 3:
        images, targets, target_lengths = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets, 0)
        target_lengths = torch.cat(target_lengths, 0)
        return images, targets, target_lengths
    
    else:
        images, file_names = zip(*batch)
        images = torch.stack(images, 0)
        return images, file_names