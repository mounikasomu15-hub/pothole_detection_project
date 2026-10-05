"""Random Forest helpers: build, save/load, and feature-table utilities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
from inference.feature_extraction import FEATURE_NAMES


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS, max_depth=config.RF_MAX_DEPTH,
        class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1)


def save_model(model, meta: dict, path=config.RF_CKPT) -> None:
    joblib.dump({"model": model, "feature_names": list(FEATURE_NAMES), "meta": meta}, path)


def load_model(path=config.RF_CKPT) -> dict:
    return joblib.load(path)


def load_labeled_table(csv_path=config.SEVERITY_CSV) -> pd.DataFrame:
    """Rows of severity_labels.csv that have a valid severity label and a non-empty mask."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run scripts/generate_feature_dataset.py first.")
    df = pd.read_csv(csv_path)
    if "severity" not in df.columns:
        raise ValueError("severity_labels.csv has no 'severity' column.")
    df["severity"] = df["severity"].astype(str).str.strip().str.capitalize()
    df = df[df["severity"].isin(config.SEVERITY_CLASSES)]
    df = df[df["area_ratio"] > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No labelled rows found. Run scripts/assign_severity_labels.py (or fill in the "
                         "'severity' column manually) before training the Random Forest.")
    return df


def split_table(df: pd.DataFrame):
    """Return (train_df, val_df, test_df).

    Uses dataset/train|val|test membership (same split as the U-Net) when the split folders
    cover the labelled images; otherwise falls back to a seeded stratified 70/15/15 split.
    """
    from sklearn.model_selection import train_test_split
    names = {s: {p.stem for p in config.SPLIT_IMAGES[s].glob("*")} if config.SPLIT_IMAGES[s].exists() else set()
             for s in config.SPLITS}
    stems = df["image_name"].astype(str).str.rsplit(".", n=1).str[0]
    masks = {s: stems.isin(names[s]) for s in config.SPLITS}
    if all(m.sum() > 0 for m in masks.values()) and sum(m.sum() for m in masks.values()) >= 0.9 * len(df):
        return df[masks["train"]], df[masks["val"]], df[masks["test"]]

    def _split(d, test_size):
        strat = d["severity"] if d["severity"].value_counts().min() >= 2 else None
        return train_test_split(d, test_size=test_size, random_state=config.RANDOM_SEED, stratify=strat)

    train_val, test = _split(df, config.TEST_RATIO)
    train, val = _split(train_val, config.VAL_RATIO / (1 - config.TEST_RATIO))
    return train, val, test
