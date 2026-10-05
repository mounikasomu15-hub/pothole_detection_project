"""Evaluate the Random Forest on the test rows; saves metrics JSON + confusion matrix."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, # type: ignore
                             precision_recall_fscore_support)

import config
from models.random_forest import load_labeled_table, load_model, split_table
from utils.helpers import setup_logging, write_json
from utils.visualization import plot_confusion_matrix

log = setup_logging("evaluate_random_forest")


def main():
    if not Path(config.RF_CKPT).exists():
        print("ERROR: Model files not found. Please train the models first (python training/train_random_forest.py).")
        sys.exit(1)
    bundle = load_model()
    try:
        _, _, test = split_table(load_labeled_table())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    if len(test) == 0:
        print("ERROR: No test rows available.")
        sys.exit(1)

    y_true = test["severity"]
    y_pred = bundle["model"].predict(test[bundle["feature_names"]])
    labels = [c for c in config.SEVERITY_CLASSES if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
        "precision_macro": pm, "recall_macro": rm, "f1_macro": fm,
        "labels": labels, "confusion_matrix": cm.tolist(), "num_test_samples": int(len(test)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "label_sources": bundle.get("meta", {}).get("label_sources"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(config.REPORTS_DIR / "rf_metrics.json", metrics)
    plot_confusion_matrix(cm, labels, config.GRAPHS_DIR / "rf_confusion_matrix.png", "Random Forest - Test Set")
    log.info("Random Forest test accuracy: %.4f (n=%d)", metrics["accuracy"], len(test))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f"Saved: {config.REPORTS_DIR / 'rf_metrics.json'}")
    print(f"Saved: {config.GRAPHS_DIR / 'rf_confusion_matrix.png'}")
    if "rule_based" in " ".join(metrics["label_sources"] or []):
        print("\nNOTE: labels are RULE-BASED (derived from the same features). High accuracy therefore shows "
              "the forest re-learned the rule - it is not evidence of real-world severity accuracy.")


if __name__ == "__main__":
    main()
