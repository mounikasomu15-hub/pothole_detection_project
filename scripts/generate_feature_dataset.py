"""Extract shape features from masks and write dataset/severity_labels.csv.

  --source gt    (default) features from the ground-truth masks in dataset/masks
  --source pred  features from U-Net PREDICTED masks (matches what the Random Forest sees at
                 inference time; needs a trained U-Net)

Existing severity labels in the CSV are preserved (matched by image_name), so manual
annotations are never lost when you regenerate features.
Severity is NOT assigned here -> run scripts/assign_severity_labels.py or label manually.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

import config
from inference.feature_extraction import EXTRA_FEATURE_NAMES, FEATURE_NAMES, extract_features
from preprocessing.preprocess_images import preprocess_image, resize_mask
from utils.helpers import find_pairs, load_image_rgb, load_mask, setup_logging

log = setup_logging("generate_feature_dataset")
FIRST = ["image_name", "area_ratio", "width", "height", "aspect_ratio", "perimeter", "circularity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["gt", "pred"], default="gt")
    args = ap.parse_args()

    pairs, _, _ = find_pairs(config.IMAGES_DIR, config.MASKS_DIR)
    if not pairs:
        print("ERROR: no image/mask pairs in dataset/images and dataset/masks."); sys.exit(1)

    segmenter = None
    if args.source == "pred":
        from inference.pothole_detector import clean_mask
        from inference.segmentation import Segmenter
        if not config.UNET_CKPT.exists():
            print("ERROR: Model files not found. Train the U-Net first to use --source pred."); sys.exit(1)
        segmenter = Segmenter()
    size = segmenter.image_size if segmenter else config.IMAGE_SIZE

    old = {}
    if config.SEVERITY_CSV.exists():
        prev = pd.read_csv(config.SEVERITY_CSV)
        for _, r in prev.iterrows():
            if isinstance(r.get("severity"), str) and r["severity"].strip():
                old[r["image_name"]] = (r["severity"], r.get("severity_source", "manual"))

    rows = []
    for ip, mp in tqdm(pairs, desc="Extracting features"):
        try:
            if segmenter:
                prob = segmenter.predict_probability(preprocess_image(load_image_rgb(ip), size))
                mask = clean_mask((prob > config.SEGMENTATION_THRESHOLD).astype("uint8"))
            else:
                mask = resize_mask(load_mask(mp), size)
            f = extract_features(mask)
        except Exception as exc:
            log.warning("skipped %s: %s", ip.name, exc); continue
        row = {"image_name": ip.name, **{k: f[k] for k in FEATURE_NAMES + EXTRA_FEATURE_NAMES
                                         if k not in ("bbox_x", "bbox_y")}}
        sev, src = old.get(ip.name, ("", ""))
        row.update({"feature_source": args.source, "severity": sev, "severity_source": src})
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = FIRST + [c for c in df.columns if c not in FIRST and c not in ("severity", "severity_source")]
    df = df[cols + ["severity", "severity_source"]]
    df.to_csv(config.SEVERITY_CSV, index=False)
    log.info("Wrote %d rows to %s (feature source: %s)", len(df), config.SEVERITY_CSV, args.source)
    log.info("Rows with a severity label: %d | empty masks (area 0): %d",
             int((df['severity'].astype(str).str.len() > 0).sum()), int((df["area_ratio"] == 0).sum()))
    print("\nNext: python scripts/assign_severity_labels.py --method rule   (or fill the 'severity' column manually)")


if __name__ == "__main__":
    main()
