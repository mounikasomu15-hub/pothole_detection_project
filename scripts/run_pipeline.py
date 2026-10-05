"""Run the full pipeline from the command line.

  python scripts/run_pipeline.py --image path\\to\\road.jpg [--voice]
  python scripts/run_pipeline.py --folder path\\to\\folder
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import config
from inference.pipeline import ModelNotFoundError, PotholePipeline
from utils.helpers import IMAGE_EXTS, append_prediction, load_image_rgb, setup_logging
from utils.voice_alert import VoiceAlert

log = setup_logging("inference")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image"); g.add_argument("--folder")
    ap.add_argument("--voice", action="store_true")
    args = ap.parse_args()
    try:
        pipe = PotholePipeline()
    except ModelNotFoundError as exc:
        print(f"ERROR: {exc}"); sys.exit(1)

    files = [Path(args.image)] if args.image else sorted(p for p in Path(args.folder).glob("*") if p.suffix.lower() in IMAGE_EXTS)
    voice = VoiceAlert(cooldown_sec=0)
    for f in files:
        try:
            res = pipe.run(load_image_rgb(f))
        except Exception as exc:
            print(f"{f.name}: could not process ({exc})"); continue
        out = config.SEGMENTED_DIR / f"{f.stem}_result.png"
        Image.fromarray(res["overlay"]).save(out)
        append_prediction(f.name, res)
        if res["detected"]:
            print(f"{f.name}: POTHOLE | severity {res['severity']} | severity-confidence {res['severity_confidence']:.2f} "
                  f"| segmentation-confidence {res['segmentation_confidence']:.2f} | speed {res['speed']} km/h | saved {out.name}")
            if args.voice:
                voice.alert(res["severity"], res["speed"], force=True)
        else:
            print(f"{f.name}: no pothole detected ({res['message']})")


if __name__ == "__main__":
    main()
