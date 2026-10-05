"""Train the Random Forest severity classifier from dataset/severity_labels.csv."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config
from inference.feature_extraction import FEATURE_NAMES
from models.random_forest import build_random_forest, load_labeled_table, save_model, split_table
from utils.helpers import set_seed, setup_logging, write_json

log = setup_logging("train_random_forest")


def main():
    set_seed()
    try:
        df = load_labeled_table()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    train, val, test = split_table(df)
    classes = sorted(train["severity"].unique())
    if len(classes) < 2:
        print("ERROR: Training data contains only one severity class. A classifier needs at least two. "
              "Check your severity labels / thresholds.")
        sys.exit(1)
    log.info("Rows -> train %d | val %d | test %d | class counts (train): %s",
             len(train), len(val), len(test), train["severity"].value_counts().to_dict())
    sources = sorted(df["severity_source"].dropna().unique().tolist()) if "severity_source" in df.columns else ["unknown"]
    log.info("Severity label source(s): %s", sources)

    model = build_random_forest()
    model.fit(train[FEATURE_NAMES], train["severity"])

    report = {"trained_at": datetime.now().isoformat(timespec="seconds"), "label_sources": sources,
              "n_train": len(train), "n_val": len(val), "n_test": len(test),
              "feature_importance": dict(zip(FEATURE_NAMES, map(float, model.feature_importances_)))}
    if len(val):
        pred = model.predict(val[FEATURE_NAMES])
        p, r, f, _ = precision_recall_fscore_support(val["severity"], pred, average="weighted", zero_division=0)
        report["validation"] = {"accuracy": accuracy_score(val["severity"], pred),
                                "precision_weighted": p, "recall_weighted": r, "f1_weighted": f}
        log.info("Validation accuracy: %.4f", report["validation"]["accuracy"])
    save_model(model, {"label_sources": sources, "trained_at": report["trained_at"]})
    write_json(config.REPORTS_DIR / "rf_training.json", report)
    log.info("Saved model to %s", config.RF_CKPT)
    print("\nNext: python training/evaluate_random_forest.py")


if __name__ == "__main__":
    main()
