"""Shape features from a binary segmentation mask.

IMPORTANT: these are 2-D, image-based measurements. A normal RGB camera does not
measure physical pothole depth, so no depth is reported. 'Severity' in this
project is an image-based estimate derived from the segmented shape/size.
Pixel measurements are in the model's working resolution (config.IMAGE_SIZE).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config

# Columns used by the Random Forest (order matters; stored inside the .pkl).
FEATURE_NAMES = ["area_ratio", "width", "height", "aspect_ratio", "perimeter",
                 "circularity", "relative_size", "solidity", "num_regions"]
# Extra descriptive features that are reported but not fed to the classifier.
EXTRA_FEATURE_NAMES = ["area", "contour_area", "width_height_ratio", "shape_irregularity",
                       "bbox_x", "bbox_y"]


def empty_features() -> dict:
    return {k: 0.0 for k in FEATURE_NAMES + EXTRA_FEATURE_NAMES}


def extract_features(mask01: np.ndarray, min_component_pixels: int = config.MIN_COMPONENT_PIXELS) -> dict:
    mask = (np.asarray(mask01) > 0).astype(np.uint8)
    h_img, w_img = mask.shape[:2]
    feats = empty_features()

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, n_lbl) if stats[i, cv2.CC_STAT_AREA] >= min_component_pixels]
    if not keep:
        return feats
    mask = np.isin(labels, keep).astype(np.uint8)

    area = float(mask.sum())
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = float(x1 - x0 + 1), float(y1 - y0 + 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    contour_area = float(sum(cv2.contourArea(c) for c in contours))
    circ_base = contour_area if contour_area > 0 else area
    circularity = float(min(1.0, 4 * np.pi * circ_base / (perimeter ** 2))) if perimeter > 0 else 0.0
    points = np.concatenate(contours, axis=0)
    hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    solidity = float(min(1.0, circ_base / hull_area)) if hull_area > 0 else 1.0

    feats.update({
        "area": area,
        "area_ratio": area / float(h_img * w_img),
        "width": bw,
        "height": bh,
        "aspect_ratio": max(bw, bh) / max(1.0, min(bw, bh)),   # always >= 1
        "width_height_ratio": bw / bh,
        "perimeter": perimeter,
        "contour_area": contour_area,
        "circularity": circularity,
        "relative_size": (bw * bh) / float(h_img * w_img),      # bounding box / image area
        "solidity": solidity,
        "shape_irregularity": 1.0 - solidity,
        "num_regions": float(len(keep)),
        "bbox_x": float(x0),
        "bbox_y": float(y0),
    })
    return feats
