import os
from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class PCBDataset(Dataset):
    """
    Dataset loader for Circuit Board images.
    Supports:
        - Paired degraded (bad) and ground-truth (good) images
        - Single folder mode for evaluation and inference
        - Automatic cropping/resizing and tensor normalization
    """
    def __init__(
        self,
        bad_dir: Optional[str] = None,
        good_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = True
    ):
        self.bad_dir = bad_dir
        self.good_dir = good_dir
        self.image_size = image_size
        self.is_train = is_train

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize(image_size),
            T.ToTensor()
        ])

        if bad_dir and os.path.exists(bad_dir):
            self.bad_files = sorted([f for f in os.listdir(bad_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
        else:
            self.bad_files = []

        if good_dir and os.path.exists(good_dir):
            self.good_files = sorted([f for f in os.listdir(good_dir) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
        else:
            self.good_files = []

    def __len__(self) -> int:
        if self.bad_files and self.good_files:
            return min(len(self.bad_files), len(self.good_files))
        elif self.bad_files:
            return len(self.bad_files)
        return len(self.good_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bad_files and self.good_files:
            bad_path = os.path.join(self.bad_dir, self.bad_files[idx])
            good_path = os.path.join(self.good_dir, self.good_files[idx])

            bad_img = Image.open(bad_path).convert("RGB")
            good_img = Image.open(good_path).convert("RGB")

            bad_tensor = self.transform(bad_img)
            good_tensor = self.transform(good_img)
            return bad_tensor, good_tensor

        elif self.bad_files:
            bad_path = os.path.join(self.bad_dir, self.bad_files[idx])
            bad_img = Image.open(bad_path).convert("RGB")
            bad_tensor = self.transform(bad_img)
            return bad_tensor, bad_tensor

        else:
            good_path = os.path.join(self.good_dir, self.good_files[idx])
            good_img = Image.open(good_path).convert("RGB")
            good_tensor = self.transform(good_img)
            return good_tensor, good_tensor


def get_data_loader(
    bad_dir: Optional[str] = None,
    good_dir: Optional[str] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    image_size: Tuple[int, int] = (256, 256)
) -> DataLoader:
    """Helper to instantiate PCB DataLoader."""
    dataset = PCBDataset(bad_dir=bad_dir, good_dir=good_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

