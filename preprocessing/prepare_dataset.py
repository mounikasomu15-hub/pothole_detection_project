"""Import a downloaded dataset into the project layout.

Takes any folders of images + masks (matched by file name stem), converts masks to clean
binary PNG (0 = background, 255 = pothole) and copies everything into dataset/images and
dataset/masks. Optionally renames files to image_0001.jpg / image_0001.png.

Usage (Windows):
  python preprocessing/prepare_dataset.py --images "C:\\path\\to\\images" --masks "C:\\path\\to\\masks" --rename
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import config
from utils.helpers import find_pairs, load_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder containing the source images")
    ap.add_argument("--masks", required=True, help="folder containing the source masks")
    ap.add_argument("--rename", action="store_true", help="rename pairs to image_0001, image_0002, ...")
    args = ap.parse_args()

    pairs, missing, orphans = find_pairs(args.images, args.masks)
    print(f"Matched pairs: {len(pairs)} | images without mask: {len(missing)} | masks without image: {len(orphans)}")
    if not pairs:
        print("ERROR: no matching image/mask names found."); sys.exit(1)
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.MASKS_DIR.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    for i, (ip, mp) in enumerate(pairs, start=1):
        stem = f"image_{i:04d}" if args.rename else ip.stem
        try:
            mask = load_mask(mp) * 255
            with Image.open(ip) as im:
                im.verify()
            Image.fromarray(mask.astype(np.uint8), mode="L").save(config.MASKS_DIR / f"{stem}.png")
            shutil.copy2(ip, config.IMAGES_DIR / f"{stem}{ip.suffix.lower()}")
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[skipped] {ip.name}: {exc}")
    print(f"Imported {done} pairs ({failed} failed) into {config.DATASET_DIR}")
    print("Next: python scripts/validate_dataset.py")


if __name__ == "__main__":
    main()
