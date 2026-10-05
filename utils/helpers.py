"""Shared helpers: seeding, device, logging, file/pair discovery, history CSV."""
import csv
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device) -> str:
    return "CUDA" if str(device).startswith("cuda") else "CPU"


def setup_logging(name: str) -> logging.Logger:
    """Console + logs/<name>.log logger (safe to call repeatedly)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(config.LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# ------------------------------------------------------------------ image IO
def load_image_rgb(path) -> np.ndarray:
    """Open an image (Unicode-safe on Windows) and return RGB uint8 HxWx3."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return np.array(img.convert("RGB"), dtype=np.uint8)


def binarize_array(arr: np.ndarray) -> np.ndarray:
    """Convert any mask-like array (0/1, 0/255, anti-aliased) to uint8 {0,1}."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    arr = arr.astype(np.int32)
    if arr.max() <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if arr.max() == 1:
        return (arr > 0).astype(np.uint8)
    return (arr > arr.max() / 2.0).astype(np.uint8)


def load_mask(path) -> np.ndarray:
    """Load a segmentation mask as uint8 {0,1}. Palette masks use raw indices."""
    with Image.open(path) as img:
        arr = np.array(img)  # palette ('P') images keep their index values
    return binarize_array(arr)


def find_pairs(images_dir, masks_dir):
    """Match images and masks by file stem (image_0001.jpg <-> image_0001.png).

    Returns (pairs, images_without_mask, masks_without_image).
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    imgs = {}
    for p in sorted(images_dir.glob("*")) if images_dir.exists() else []:
        if p.suffix.lower() in IMAGE_EXTS:
            imgs.setdefault(p.stem, p)
    masks = {}
    for p in sorted(masks_dir.glob("*")) if masks_dir.exists() else []:
        if p.suffix.lower() in MASK_EXTS:
            if p.stem not in masks or p.suffix.lower() == ".png":
                masks[p.stem] = p
    pairs = [(imgs[s], masks[s]) for s in sorted(imgs) if s in masks]
    missing_masks = sorted(s for s in imgs if s not in masks)
    orphan_masks = sorted(s for s in masks if s not in imgs)
    return pairs, missing_masks, orphan_masks


# ------------------------------------------------------------------ json
def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ------------------------------------------------------------ history CSV
HISTORY_COLUMNS = ["timestamp", "image_name", "pothole_detected", "severity",
                   "confidence", "pothole_area", "recommended_speed"]


def append_prediction(image_name: str, result: dict) -> None:
    """Append one row to outputs/predictions/predictions.csv."""
    new_file = not config.PREDICTIONS_CSV.exists()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "pothole_detected": bool(result["detected"]),
        "severity": result.get("severity") or "",
        "confidence": "" if result.get("severity_confidence") is None
        else round(float(result["severity_confidence"]), 4),
        "pothole_area": int(result.get("area_px", 0)),
        "recommended_speed": "" if result.get("speed") is None else int(result["speed"]),
    }
    with open(config.PREDICTIONS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def read_history(n: int = 20):
    """Return the last n history rows as a DataFrame (or None)."""
    if not config.PREDICTIONS_CSV.exists():
        return None
    import pandas as pd
    try:
        return pd.read_csv(config.PREDICTIONS_CSV).tail(n).iloc[::-1].reset_index(drop=True)
    except Exception:
        return None
