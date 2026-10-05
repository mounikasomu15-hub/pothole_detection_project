"""Plotting / drawing helpers (matplotlib + OpenCV)."""
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # figures are written to files (outputs/graphs)
import matplotlib.pyplot as plt  # noqa: E402


def overlay_mask(image_rgb, mask01, color=(255, 0, 0), alpha=0.45):
    out = image_rgb.copy()
    m = mask01.astype(bool)
    if m.any():
        col = np.array(color, dtype=np.float32)
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + col * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def draw_bbox(image_rgb, bbox, color=(0, 255, 0), thickness=2, label=None):
    """bbox = (x, y, w, h) in pixel coordinates of image_rgb."""
    out = image_rgb.copy()
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    if label:
        cv2.putText(out, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def mask_to_rgb(mask01):
    return np.repeat((mask01.astype(np.uint8) * 255)[..., None], 3, axis=2)


def plot_training_history(history: dict, path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, history["train_loss"], label="Train loss")
    ax[0].plot(ep, history["val_loss"], label="Validation loss")
    ax[0].set_title("Loss (BCE + Dice)"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, history["val_iou"], color="tab:green"); ax[1].set_title("Validation IoU"); ax[1].set_xlabel("Epoch")
    ax[2].plot(ep, history["val_dice"], color="tab:red"); ax[2].set_title("Validation Dice"); ax[2].set_xlabel("Epoch")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm, labels, path, title="Confusion Matrix"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_sample_grid(rows, path):
    """rows: list of (image_rgb, gt_mask01, pred_mask01). Saves Original | GT | Pred | Overlay."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), squeeze=False)
    for i, (img, gt, pred) in enumerate(rows):
        panels = [(img, "Original"), (gt, "Ground truth"), (pred, "Predicted mask"),
                  (overlay_mask(img, pred), "Overlay")]
        for j, (data, title) in enumerate(panels):
            axes[i][j].imshow(data, cmap="gray" if data.ndim == 2 else None)
            axes[i][j].set_title(title, fontsize=9)
            axes[i][j].axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
