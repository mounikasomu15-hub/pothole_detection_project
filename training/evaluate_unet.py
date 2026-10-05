"""Evaluate the U-Net on the held-out test split. Metrics are computed from real predictions."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

import config
from inference.segmentation import Segmenter
from preprocessing.dataset import PotholeDataset
from preprocessing.preprocess_images import denormalize_image
from utils.helpers import device_name, get_device, set_seed, setup_logging, write_json
from utils.metrics import SegmentationMeter
from utils.visualization import save_sample_grid

log = setup_logging("evaluate_unet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6, help="number of qualitative examples to save")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    set_seed()
    device = get_device()
    print(f"Using device: {device_name(device)}")
    if not Path(config.UNET_CKPT).exists():
        print("ERROR: Model files not found. Please train the models first (python training/train_unet.py).")
        sys.exit(1)
    seg = Segmenter(config.UNET_CKPT, device)
    try:
        ds = PotholeDataset(config.SPLIT_IMAGES[args.split], config.SPLIT_MASKS[args.split], size=seg.image_size)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    meter = SegmentationMeter()
    sample_idx = set(np.linspace(0, len(ds) - 1, min(args.samples, len(ds)), dtype=int).tolist())
    rows = []
    for i in tqdm(range(len(ds)), desc=f"Evaluating on {args.split}"):
        x, y = ds[i]
        prob = seg.predict_probability(x.unsqueeze(0))
        pred = (prob > config.SEGMENTATION_THRESHOLD).astype(np.uint8)
        meter.update(torch.from_numpy(pred), y[0] > 0.5)
        if i in sample_idx:
            rows.append((denormalize_image(x.numpy()), y[0].numpy().astype(np.uint8), pred))

    m = meter.compute()
    m.update({"num_images": len(ds), "split": args.split, "threshold": config.SEGMENTATION_THRESHOLD,
              "image_size": seg.image_size, "evaluated_at": datetime.now().isoformat(timespec="seconds")})
    write_json(config.REPORTS_DIR / "unet_metrics.json", m)
    save_sample_grid(rows, config.GRAPHS_DIR / "unet_samples.png")

    log.info("U-Net results on %d %s images:", len(ds), args.split)
    for k in ("iou", "dice", "precision", "recall", "f1", "pixel_accuracy"):
        log.info("  %-15s %.4f", k, m[k])
    print(f"\nSaved: {config.REPORTS_DIR / 'unet_metrics.json'}\nSaved: {config.GRAPHS_DIR / 'unet_samples.png'}")


if __name__ == "__main__":
    main()
