"""Segmentation metrics computed from real predictions (nothing hard-coded)."""
import torch

EPS = 1e-7


def confusion_counts(pred: torch.Tensor, target: torch.Tensor):
    """pred/target: binary tensors of identical shape. Returns (tp, fp, fn, tn) ints."""
    p, t = pred.bool(), target.bool()
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    return {
        "iou": tp / (tp + fp + fn + EPS),
        "dice": 2 * tp / (2 * tp + fp + fn + EPS),
        "precision": tp / (tp + fp + EPS),
        "recall": tp / (tp + fn + EPS),
        "f1": 2 * tp / (2 * tp + fp + fn + EPS),
        "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + EPS),
    }


class SegmentationMeter:
    """Accumulates pixel counts over a whole dataset (micro-averaged metrics)."""

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred, target):
        a, b, c, d = confusion_counts(pred, target)
        self.tp += a; self.fp += b; self.fn += c; self.tn += d

    def compute(self) -> dict:
        return metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
