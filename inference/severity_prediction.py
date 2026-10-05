"""Random Forest severity prediction with predict_proba() confidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from models.random_forest import load_model


class SeverityPredictor:
    def __init__(self, path=config.RF_CKPT):
        bundle = load_model(path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]
        self.meta = bundle.get("meta", {})

    def predict(self, features: dict):
        """Returns (severity_label, confidence, {label: probability})."""
        row = pd.DataFrame([[float(features[f]) for f in self.feature_names]], columns=self.feature_names)
        proba = self.model.predict_proba(row)[0]
        classes = [str(c) for c in self.model.classes_]
        idx = int(proba.argmax())
        return classes[idx], float(proba[idx]), {c: float(p) for c, p in zip(classes, proba)}
