"""Validate dataset/images and dataset/masks. Usage: python scripts/validate_dataset.py"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import config
from utils.helpers import IMAGE_EXTS, MASK_EXTS, write_json


def duplicates(folder: Path, exts):
    files = [p for p in folder.glob("*") if p.suffix.lower() in exts] if folder.exists() else []
    counts = Counter(p.stem for p in files)
    return sorted(s for s, c in counts.items() if c > 1), files


def main() -> int:
    img_dir, mask_dir = config.IMAGES_DIR, config.MASKS_DIR
    print(f"Images folder: {img_dir}\nMasks folder : {mask_dir}\n")
    if not img_dir.exists() or not mask_dir.exists():
        print("ERROR: dataset/images or dataset/masks folder is missing.")
        return 1
    dup_img, img_files = duplicates(img_dir, IMAGE_EXTS)
    dup_mask, mask_files = duplicates(mask_dir, MASK_EXTS)
    if not img_files:
        print("ERROR: dataset/images is empty. See README.md -> 'Dataset preparation'.")
        return 1

    img_by_stem = {p.stem: p for p in sorted(img_files)}
    mask_by_stem = {}
    for p in sorted(mask_files):
        if p.stem not in mask_by_stem or p.suffix.lower() == ".png":
            mask_by_stem[p.stem] = p
    missing_masks = sorted(set(img_by_stem) - set(mask_by_stem))
    orphan_masks = sorted(set(mask_by_stem) - set(img_by_stem))

    corrupt_images, corrupt_masks, size_mismatch, bad_values = [], [], [], []
    empty_masks, nonbinary, coverage, sizes, matched = [], [], [], Counter(), 0
    for stem in sorted(set(img_by_stem) & set(mask_by_stem)):
        ip, mp = img_by_stem[stem], mask_by_stem[stem]
        try:
            with Image.open(ip) as im:
                im.verify()
            with Image.open(ip) as im:
                isize = im.size
        except Exception:
            corrupt_images.append(ip.name); continue
        try:
            with Image.open(mp) as mm:
                mm.verify()
            with Image.open(mp) as mm:
                marr = np.array(mm)
                msize = mm.size
        except Exception:
            corrupt_masks.append(mp.name); continue
        matched += 1
        sizes[isize] += 1
        if isize != msize:
            size_mismatch.append(f"{ip.name} {isize} vs {mp.name} {msize}")
        if marr.ndim == 3:
            marr = marr.max(axis=2)
        vals = np.unique(marr)
        if not (set(vals.tolist()) <= {0, 1} or set(vals.tolist()) <= {0, 255}):
            if len(vals) > 8:
                bad_values.append(f"{mp.name}: {len(vals)} distinct values (looks like a photo, not a mask)")
            else:
                nonbinary.append(mp.name)   # e.g. JPEG artefacts: will be thresholded automatically
        if marr.max() == 0:
            empty_masks.append(mp.name)
        else:
            coverage.append(float((marr > marr.max() / 2).mean()))

    errors = (len(missing_masks) + len(corrupt_images) + len(corrupt_masks) + len(size_mismatch)
              + len(bad_values))
    def show(title, items, limit=10):
        if items:
            print(f"[!] {title}: {len(items)}")
            for it in items[:limit]:
                print("     -", it)
            if len(items) > limit:
                print(f"     ... and {len(items) - limit} more")

    show("Duplicate image filenames (same name, different extension)", dup_img)
    show("Duplicate mask filenames", dup_mask)
    show("Images WITHOUT a matching mask", missing_masks)
    show("Masks WITHOUT a matching image", orphan_masks)
    show("Corrupted images", corrupt_images)
    show("Corrupted masks", corrupt_masks)
    show("Image/mask size mismatch", size_mismatch)
    show("Masks with invalid values", bad_values)
    show("Masks with non-binary values (will be thresholded automatically)", nonbinary)
    show("Empty masks (no pothole pixels) - fine for negatives, otherwise check", empty_masks)

    print("\n================ DATASET STATISTICS ================")
    print(f"Image files            : {len(img_files)}")
    print(f"Mask files             : {len(mask_files)}")
    print(f"Valid matched pairs    : {matched - len(size_mismatch) - len(bad_values)}")
    print(f"Target dataset size    : {config.TARGET_DATASET_SIZE} (found {len(img_by_stem)} images)")
    if coverage:
        print(f"Mean pothole coverage  : {100 * np.mean(coverage):.2f}% of pixels "
              f"(min {100 * min(coverage):.2f}%, max {100 * max(coverage):.2f}%)")
    if sizes:
        print("Most common image sizes: " + ", ".join(f"{w}x{h} ({c})" for (w, h), c in sizes.most_common(3)))
    print("====================================================")
    write_json(config.REPORTS_DIR / "dataset_validation.json", {
        "images": len(img_files), "masks": len(mask_files), "matched": matched, "errors": errors,
        "missing_masks": missing_masks, "orphan_masks": orphan_masks, "corrupt_images": corrupt_images,
        "corrupt_masks": corrupt_masks, "duplicates": dup_img + dup_mask, "empty_masks": len(empty_masks)})
    if errors or dup_img or dup_mask:
        print("RESULT: problems found - fix them (or remove the listed files) before training.")
        return 1
    print("RESULT: dataset is valid. Next: python preprocessing/split_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
