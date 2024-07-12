"""

Author: Andreas Rössler
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

xception_default_data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    ),
}


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(self.root_dir)

        self.data = []  # List to store (image_path, label) pairs

        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            if not os.path.isdir(class_path):  # Check if it's a directory
                continue
            label = 0 if class_dir == "original" else 1  # 0 for original, 1 for fake

            for file_name in os.listdir(class_path):
                self.data.append((os.path.join(class_path, file_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class CustomImageDatasetFromFreq(Dataset):
    def __init__(self, root_dir, data_list_from_freq, transform=None):
        self.root_dir = root_dir + "freq/"
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.data_list_from_freq = np.load(data_list_from_freq, allow_pickle=True)

        self.data = []  # List to store (image_path, label) pairs

        data_set_from_freq = {Path(path).name for path in self.data_list_from_freq}
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            if not os.path.isdir(class_path):  # Check if it's a directory
                continue
            label = 0 if class_dir == "A_ffhq" else 1  # 0 for original, 1 for fake
            class_path_contents = set(os.listdir(class_path))

            # Convert data_list_from_freq to a set for faster lookups
            # Use set intersection for efficient matching
            matching_files = data_set_from_freq.intersection(class_path_contents)
            # Use a list comprehension for creating the data list
            self.data.extend(
                (os.path.join(class_path, file_name), label, file_name)
                for file_name in matching_files
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label, path = self.data[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label  # , path
