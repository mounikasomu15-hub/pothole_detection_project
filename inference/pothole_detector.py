"""Decide whether a pothole exists in a predicted mask."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config


def clean_mask(mask01: np.ndarray, min_pixels: int = config.MIN_COMPONENT_PIXELS) -> np.ndarray:
    """Remove tiny connected regions (segmentation noise)."""
    mask = (np.asarray(mask01) > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_pixels]
    if not keep:
        return np.zeros_like(mask)
    return np.isin(labels, keep).astype(np.uint8)


def detect_pothole(mask01: np.ndarray, min_area_ratio: float = config.MIN_AREA_RATIO) -> bool:
    mask = np.asarray(mask01) > 0
    if mask.size == 0 or not mask.any():
        return False
    return (mask.sum() / mask.size) >= min_area_ratio
