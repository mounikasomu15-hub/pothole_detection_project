"""Create / check severity labels in dataset/severity_labels.csv.

IMPORTANT - be honest about where labels come from. This script does NOT create
ground-truth labels from a physical measurement. Options:

  --method rule        RULE-BASED labels from area_ratio using thresholds in config.py
                       (RULE_AREA_RATIO_LOW_MAX / RULE_AREA_RATIO_MEDIUM_MAX).
                       severity_source = 'rule_based_area_ratio'
  --method percentile  RULE-BASED labels from area_ratio terciles of YOUR dataset
                       (bottom 1/3 = Low, middle = Medium, top 1/3 = High).
                       severity_source = 'rule_based_percentile'
  --method check       only validate manually typed labels (Low / Medium / High) and print counts.

Manual labels already present are kept unless you pass --overwrite.
Because rule-based labels are derived from the same features the Random Forest uses, the
forest will largely re-learn the rule. Manual (expert) labels are scientifically stronger.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from utils.helpers import write_json


def label_rule(a):
    if a < config.RULE_AREA_RATIO_LOW_MAX:
        return "Low"
    return "Medium" if a < config.RULE_AREA_RATIO_MEDIUM_MAX else "High"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rule", "percentile", "check"], default="rule")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not config.SEVERITY_CSV.exists():
        print("ERROR: dataset/severity_labels.csv not found. Run scripts/generate_feature_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(config.SEVERITY_CSV)
    df["severity"] = df["severity"].fillna("").astype(str).str.strip().str.capitalize()
    df["severity_source"] = df.get("severity_source", "").fillna("") if "severity_source" in df else ""

    if args.method != "check":
        valid = df["area_ratio"] > 0
        target = valid if args.overwrite else (valid & (df["severity"] == ""))
        if args.method == "rule":
            df.loc[target, "severity"] = df.loc[target, "area_ratio"].map(label_rule)
            df.loc[target, "severity_source"] = "rule_based_area_ratio"
        else:
            q1, q2 = df.loc[valid, "area_ratio"].quantile([1 / 3, 2 / 3])
            df.loc[target, "severity"] = df.loc[target, "area_ratio"].map(
                lambda a: "Low" if a <= q1 else ("Medium" if a <= q2 else "High"))
            df.loc[target, "severity_source"] = "rule_based_percentile"
            write_json(config.REPORTS_DIR / "severity_thresholds.json", {"q33": float(q1), "q66": float(q2)})
        df.to_csv(config.SEVERITY_CSV, index=False)
        print(f"Labelled {int(target.sum())} rows using method '{args.method}'.")

    bad = df[(df["severity"] != "") & (~df["severity"].isin(config.SEVERITY_CLASSES))]
    if len(bad):
        print(f"[!] {len(bad)} rows have an invalid severity (allowed: {config.SEVERITY_CLASSES}):")
        print(bad[["image_name", "severity"]].head(10).to_string(index=False))
    print("\nSeverity counts:\n", df[df["severity"] != ""]["severity"].value_counts().to_string())
    print("\nLabel sources:\n", df[df["severity"] != ""]["severity_source"].replace("", "manual/unspecified")
          .value_counts().to_string())
    print("\nNext: python training/train_random_forest.py")


if __name__ == "__main__":
    main()
