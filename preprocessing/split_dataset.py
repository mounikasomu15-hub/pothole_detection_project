"""Split dataset/images + dataset/masks into train/val/test (70/15/15 by default).

Usage: python preprocessing/split_dataset.py [--clean] [--seed 42]
Files are COPIED, so dataset/images and dataset/masks stay untouched.
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

import config
from utils.helpers import find_pairs


def readable(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=config.TRAIN_RATIO)
    ap.add_argument("--val", type=float, default=config.VAL_RATIO)
    ap.add_argument("--test", type=float, default=config.TEST_RATIO)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--clean", action="store_true", help="delete existing split folders' contents first")
    args = ap.parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print("ERROR: train + val + test ratios must sum to 1.0"); sys.exit(1)

    pairs, missing, orphans = find_pairs(config.IMAGES_DIR, config.MASKS_DIR)
    if missing:
        print(f"[warning] {len(missing)} images have no mask and are excluded.")
    if not pairs:
        print("ERROR: no image/mask pairs found in dataset/images and dataset/masks. See README.md."); sys.exit(1)

    existing = [p for s in config.SPLITS for d in (config.SPLIT_IMAGES[s], config.SPLIT_MASKS[s])
                for p in d.glob("*") if p.name != ".gitkeep"]
    if existing and not args.clean:
        print(f"ERROR: split folders already contain {len(existing)} files. Re-run with --clean to rebuild them.")
        sys.exit(1)
    for p in existing:
        p.unlink()

    good = [(i, m) for i, m in pairs if readable(i) and readable(m)]
    if len(good) < len(pairs):
        print(f"[warning] skipped {len(pairs) - len(good)} corrupted pairs (run scripts/validate_dataset.py).")
    random.Random(args.seed).shuffle(good)
    n = len(good)
    n_train, n_val = int(n * args.train), int(n * args.val)
    parts = {"train": good[:n_train], "val": good[n_train:n_train + n_val], "test": good[n_train + n_val:]}

    rows = []
    for split, items in parts.items():
        config.SPLIT_IMAGES[split].mkdir(parents=True, exist_ok=True)
        config.SPLIT_MASKS[split].mkdir(parents=True, exist_ok=True)
        for ip, mp in items:
            shutil.copy2(ip, config.SPLIT_IMAGES[split] / ip.name)
            shutil.copy2(mp, config.SPLIT_MASKS[split] / mp.name)
            rows.append((split, ip.name, mp.name))
        print(f"{split:5s}: {len(items)} pairs")
    with open(config.REPORTS_DIR / "dataset_split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["split", "image", "mask"]); w.writerows(rows)
    print(f"Total {n} pairs split (seed={args.seed}). Manifest: {config.REPORTS_DIR / 'dataset_split.csv'}")
    print("Next: python training/train_unet.py")


if __name__ == "__main__":
    main()
