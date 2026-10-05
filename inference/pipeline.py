"""End-to-end pipeline:
image -> preprocess -> U-Net -> mask -> detection -> features -> Random Forest
      -> severity (+confidence) -> safe speed.
Voice alerts are triggered by the caller (Streamlit app / scripts) via utils.voice_alert.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from inference.feature_extraction import extract_features
from inference.pothole_detector import clean_mask, detect_pothole
from inference.segmentation import Segmenter, segmentation_confidence
from inference.severity_prediction import SeverityPredictor
from preprocessing.preprocess_images import preprocess_image
from utils.helpers import get_device
from utils.speed_recommendation import get_explanation, get_safe_speed
from utils.visualization import draw_bbox, overlay_mask


class ModelNotFoundError(Exception):
    """Raised when the trained model files are missing."""


def missing_model_files():
    return [str(p) for p in (config.UNET_CKPT, config.RF_CKPT) if not Path(p).exists()]


class PotholePipeline:
    def __init__(self, unet_path=config.UNET_CKPT, rf_path=config.RF_CKPT, device=None):
        missing = [str(p) for p in (unet_path, rf_path) if not Path(p).exists()]
        if missing:
            raise ModelNotFoundError("Model files not found. Please train the models first. Missing: "
                                     + ", ".join(missing))
        self.device = device or get_device()
        self.segmenter = Segmenter(unet_path, self.device)
        self.severity_model = SeverityPredictor(rf_path)

    # step functions (names mirror the project description)
    def preprocess(self, image_rgb):
        return preprocess_image(image_rgb, self.segmenter.image_size)

    def unet_predict(self, tensor, seg_threshold):
        prob = self.segmenter.predict_probability(tensor)
        mask = clean_mask((prob > seg_threshold).astype(np.uint8))
        return mask, prob

    def run(self, image_rgb, seg_threshold=config.SEGMENTATION_THRESHOLD,
            conf_threshold=config.CONFIDENCE_THRESHOLD, min_area_ratio=config.MIN_AREA_RATIO) -> dict:
        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("Input must be an RGB uint8 image (HxWx3).")
        H, W = image_rgb.shape[:2]

        tensor = self.preprocess(image_rgb)
        mask_small, prob_small = self.unet_predict(tensor, seg_threshold)
        seg_conf = segmentation_confidence(prob_small, mask_small)
        detected = detect_pothole(mask_small, min_area_ratio) and seg_conf >= conf_threshold

        # full-resolution mask for display (probability resized, then thresholded)
        prob_full = cv2.resize(prob_small, (W, H), interpolation=cv2.INTER_LINEAR)
        keep = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        mask_full = ((prob_full > seg_threshold) & keep).astype(np.uint8)
        if not detected:
            mask_full[:] = 0

        result = {
            "detected": bool(detected), "mask": mask_full, "mask_small": mask_small,
            "probability": prob_full, "segmentation_confidence": seg_conf,
            "features": None, "severity": None, "confidence": None, "severity_confidence": None,
            "severity_probabilities": None, "speed": None, "explanation": None, "bbox": None,
            "area_px": 0, "area_ratio": 0.0, "overlay": image_rgb.copy(),
            "message": "No pothole detected. The road surface appears clear.",
        }
        if not detected:
            if mask_small.any():
                result["message"] = ("A very small / low-confidence region was found but it is below the "
                                     "detection thresholds, so no pothole is reported.")
            return result

        feats = extract_features(mask_small)
        severity, sev_conf, probs = self.severity_model.predict(feats)
        speed = get_safe_speed(severity)
        sx, sy = W / mask_small.shape[1], H / mask_small.shape[0]
        bbox = (feats["bbox_x"] * sx, feats["bbox_y"] * sy, feats["width"] * sx, feats["height"] * sy)
        overlay = overlay_mask(image_rgb, mask_full)
        overlay = draw_bbox(overlay, bbox, label=f"{severity} | {speed} km/h")
        result.update({
            "features": feats, "severity": severity, "confidence": sev_conf,
            "severity_confidence": sev_conf, "severity_probabilities": probs, "speed": speed,
            "explanation": get_explanation(severity), "bbox": bbox, "overlay": overlay,
            "area_px": int(mask_full.sum()), "area_ratio": float(mask_full.mean()),
            "message": "Pothole detected.",
        })
        return result


_default_pipeline = None


def run_pipeline(image_rgb, **kwargs) -> dict:
    """Convenience wrapper: loads the models once, then runs the full pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = PotholePipeline()
    return _default_pipeline.run(image_rgb, **kwargs)
