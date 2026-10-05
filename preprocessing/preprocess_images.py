"""Preprocessing shared by training AND inference (single source of truth).

Images: resize (INTER_AREA) -> scale to [0,1] -> ImageNet mean/std -> CHW tensor.
Masks : resize with INTER_NEAREST so they stay strictly binary {0,1}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from utils.helpers import binarize_array

_MEAN = np.array(config.IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(config.IMAGENET_STD, dtype=np.float32)


def resize_image(image_rgb: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask01: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Nearest-neighbour resize: never creates grey (interpolated) mask values."""
    m = cv2.resize(mask01.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return binarize_array(m)


def normalize_image(image_rgb_resized: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 3xHxW, normalised."""
    x = image_rgb_resized.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def denormalize_image(chw: np.ndarray) -> np.ndarray:
    """float 3xHxW (normalised) -> uint8 HxWx3 (for visualisation)."""
    x = np.transpose(chw, (1, 2, 0)) * _STD + _MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def preprocess_image(image_rgb: np.ndarray, size: int = config.IMAGE_SIZE):
    """Full inference preprocessing. Returns a torch tensor of shape (1, 3, size, size)."""
    import torch
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image of shape HxWx3.")
    x = normalize_image(resize_image(image_rgb, size))
    return torch.from_numpy(x).unsqueeze(0)


if __name__ == "__main__":
    dummy = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    print("Preprocessed tensor shape:", tuple(preprocess_image(dummy).shape))
