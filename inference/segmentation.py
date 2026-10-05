"""Loads the trained U-Net and produces pothole probability maps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import config
from models.unet import UNet


class Segmenter:
    def __init__(self, ckpt_path=config.UNET_CKPT, device=None):
        from utils.helpers import get_device
        self.device = device or get_device()
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(f"Could not read U-Net checkpoint '{ckpt_path}': {exc}") from exc
        # Use the settings the model was trained with (keeps inference == training).
        self.image_size = int(ckpt.get("image_size", config.IMAGE_SIZE))
        base = int(ckpt.get("base_channels", config.UNET_BASE_CHANNELS))
        self.model = UNet(3, 1, base).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.meta = {k: ckpt.get(k) for k in ("epoch", "val_dice", "val_iou")}

    @torch.no_grad()
    def predict_probability(self, tensor: torch.Tensor) -> np.ndarray:
        """tensor (1,3,S,S) -> probability map (S,S) float32 in [0,1]."""
        logits = self.model(tensor.to(self.device))
        return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)


def segmentation_confidence(prob: np.ndarray, mask01: np.ndarray) -> float:
    """Mean U-Net probability over pixels predicted as pothole (0 if mask is empty)."""
    m = mask01 > 0
    return float(prob[m].mean()) if m.any() else 0.0
