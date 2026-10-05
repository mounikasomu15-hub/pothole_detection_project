"""PyTorch Dataset with spatially-synchronised augmentation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config
from preprocessing.preprocess_images import normalize_image, resize_image, resize_mask
from utils.helpers import find_pairs, load_image_rgb, load_mask


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Geometric transforms use the SAME parameters for image and mask.
    The mask uses nearest-neighbour interpolation. Photometric changes touch the image only."""
    if rng.random() < 0.5:                                   # horizontal flip
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:                                   # small rotation (+-10 deg)
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-10, 10)), 1.0)
        image = cv2.warpAffine(np.ascontiguousarray(image), M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.8:                                   # brightness / contrast (image only)
        alpha, beta = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class PotholeDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=config.IMAGE_SIZE, augment_data=False):
        self.pairs, missing, _ = find_pairs(images_dir, masks_dir)
        if missing:
            print(f"[warning] {len(missing)} images in {images_dir} have no mask and are skipped.")
        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found in '{images_dir}' and '{masks_dir}'. "
                "Place your dataset in dataset/images and dataset/masks, then run "
                "scripts/validate_dataset.py and preprocessing/split_dataset.py.")
        self.size, self.augment_data = size, augment_data
        self.rng = np.random.default_rng(config.RANDOM_SEED)

    def __len__(self):
        return len(self.pairs)

    def load_raw(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = resize_image(load_image_rgb(img_path), self.size)
        mask = resize_mask(load_mask(mask_path), self.size)
        return image, mask

    def __getitem__(self, idx):
        try:
            image, mask = self.load_raw(idx)
        except Exception as exc:  # corrupted file -> use a neighbour instead of crashing training
            print(f"[warning] could not read {self.pairs[idx][0].name}: {exc}")
            image, mask = self.load_raw((idx + 1) % len(self.pairs))
        if self.augment_data:
            seed = int(torch.randint(0, 2**31 - 1, (1,)))   # differs per worker/epoch
            image, mask = augment(image, mask, np.random.default_rng(seed))
        x = torch.from_numpy(normalize_image(image))
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y
