"""Train the U-Net. Usage: python training/train_unet.py [--epochs N] [--batch-size N] [--lr X]"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models.unet import UNet
from preprocessing.dataset import PotholeDataset
from utils.helpers import device_name, get_device, set_seed, setup_logging, write_json
from utils.metrics import SegmentationMeter
from utils.visualization import plot_training_history

log = setup_logging("train_unet")


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return self.bce(logits, target) + dice_loss(logits, target)


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    meter, total, n = SegmentationMeter(), 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            if not train:
                meter.update(torch.sigmoid(logits) > config.SEGMENTATION_THRESHOLD, y > 0.5)
    return total / max(n, 1), (None if train else meter.compute())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    print(f"Using device: {device_name(device)}")
    log.info("Using device: %s", device_name(device))

    try:
        train_ds = PotholeDataset(config.SPLIT_IMAGES["train"], config.SPLIT_MASKS["train"],
                                  augment_data=config.USE_AUGMENTATION)
        val_ds = PotholeDataset(config.SPLIT_IMAGES["val"], config.SPLIT_MASKS["val"])
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\nRun: python scripts/validate_dataset.py  then  python preprocessing/split_dataset.py")
        sys.exit(1)
    log.info("Train images: %d | Validation images: %d", len(train_ds), len(val_ds))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=pin)

    model = UNet(3, 1, config.UNET_BASE_CHANNELS).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_dice, bad_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["dice"])
        for k, v in zip(history, (tr_loss, va_loss, va["iou"], va["dice"])):
            history[k].append(float(v))
        log.info("Epoch %d/%d | train_loss %.4f | val_loss %.4f | val_IoU %.4f | val_Dice %.4f | %.0fs",
                 epoch, args.epochs, tr_loss, va_loss, va["iou"], va["dice"], time.time() - t0)

        write_json(config.REPORTS_DIR / "unet_history.json", history)   # saved every epoch
        plot_training_history(history, config.GRAPHS_DIR / "unet_training.png")

        if va["dice"] > best_dice:
            best_dice, bad_epochs = va["dice"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_dice": va["dice"],
                        "val_iou": va["iou"], "image_size": config.IMAGE_SIZE,
                        "base_channels": config.UNET_BASE_CHANNELS}, config.UNET_CKPT)
            log.info("  -> saved best model to %s", config.UNET_CKPT)
        else:
            bad_epochs += 1
            if bad_epochs >= config.EARLY_STOP_PATIENCE:
                log.info("Early stopping (no val Dice improvement for %d epochs).", bad_epochs)
                break

    log.info("Training finished. Best validation Dice: %.4f", best_dice)
    print("\nNext: python training/evaluate_unet.py")


if __name__ == "__main__":
    main()
