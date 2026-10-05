"""Quick installation check using SYNTHETIC data in a temporary folder.

It only verifies that every script runs end to end (2 training epochs). The results are
meaningless and are discarded; your real dataset/checkpoints are never touched.
Usage: python scripts/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_synthetic(work: Path, n=60):
    rng = np.random.default_rng(0)
    (work / "dataset" / "images").mkdir(parents=True); (work / "dataset" / "masks").mkdir(parents=True)
    for i in range(1, n + 1):
        img = np.full((160, 200, 3), 110, np.uint8) + rng.integers(0, 25, (160, 200, 3), dtype=np.uint8)
        mask = np.zeros((160, 200), np.uint8)
        if i % 5:  # 1 in 5 images has no pothole
            c = (int(rng.integers(40, 160)), int(rng.integers(40, 120)))
            ax = (int(rng.integers(8, 45)), int(rng.integers(6, 30)))
            cv2.ellipse(mask, c, ax, 0, 0, 360, 255, -1)
            cv2.ellipse(img, c, ax, 0, 0, 360, (40, 40, 40), -1)
        cv2.imwrite(str(work / "dataset" / "images" / f"image_{i:04d}.jpg"), img)
        cv2.imwrite(str(work / "dataset" / "masks" / f"image_{i:04d}.png"), mask)


def run(args, env):
    print("\n>>>", " ".join(args))
    r = subprocess.run([sys.executable, *args], cwd=ROOT, env=env)
    if r.returncode != 0:
        print("SMOKE TEST FAILED at:", " ".join(args)); sys.exit(1)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        make_synthetic(work)
        env = {**os.environ, "POTHOLE_WORKDIR": str(work), "PYTHONIOENCODING": "utf-8"}
        for step in (["scripts/validate_dataset.py"], ["preprocessing/split_dataset.py"],
                     ["training/train_unet.py", "--epochs", "2"], ["training/evaluate_unet.py"],
                     ["scripts/generate_feature_dataset.py"], ["scripts/assign_severity_labels.py", "--method", "percentile"],
                     ["training/train_random_forest.py"], ["training/evaluate_random_forest.py"],
                     ["scripts/run_pipeline.py", "--image", str(work / "dataset" / "images" / "image_0001.jpg")]):
            run(step, env)
    print("\nSMOKE TEST PASSED: installation and code paths work. (Synthetic results are meaningless.)")


if __name__ == "__main__":
    main()
