"""Central configuration for the pothole detection project.

Every path and tunable value lives here. Set the environment variable
POTHOLE_WORKDIR to relocate dataset/, checkpoints/, outputs/ and logs/
(used by scripts/smoke_test.py so that your real data is never touched).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("POTHOLE_WORKDIR", PROJECT_ROOT)).resolve()

# ----------------------------------------------------------------- paths
DATASET_DIR = WORK_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
SPLITS = ("train", "val", "test")
SPLIT_IMAGES = {s: DATASET_DIR / s / "images" for s in SPLITS}
SPLIT_MASKS = {s: DATASET_DIR / s / "masks" for s in SPLITS}
SEVERITY_CSV = DATASET_DIR / "severity_labels.csv"

CHECKPOINT_DIR = WORK_DIR / "checkpoints"
UNET_CKPT = CHECKPOINT_DIR / "unet_best.pth"
RF_CKPT = CHECKPOINT_DIR / "random_forest.pkl"

OUTPUT_DIR = WORK_DIR / "outputs"
SEGMENTED_DIR = OUTPUT_DIR / "segmented"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
GRAPHS_DIR = OUTPUT_DIR / "graphs"
REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_CSV = PREDICTIONS_DIR / "predictions.csv"
LOG_DIR = WORK_DIR / "logs"

for _d in (CHECKPOINT_DIR, SEGMENTED_DIR, PREDICTIONS_DIR, GRAPHS_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------- reproducibility
RANDOM_SEED = 42

# ------------------------------------------------------------ U-Net / training
IMAGE_SIZE = 128            # images AND masks are resized to IMAGE_SIZE x IMAGE_SIZE
UNET_BASE_CHANNELS = 32     # 16 = lighter/faster, 64 = heavier
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10    # stop if validation Dice does not improve for N epochs
NUM_WORKERS = 0 if os.name == "nt" else 2   # 0 is the safest on Windows
SEGMENTATION_THRESHOLD = 0.5   # probability above which a pixel is "pothole"
USE_AUGMENTATION = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # fixed normalisation constants
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------ dataset split
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
TARGET_DATASET_SIZE = 5000   # only used for an informational message

# ------------------------------------------------------------ detection
MIN_AREA_RATIO = 0.002       # pothole area / image area below this => NOT DETECTED
MIN_COMPONENT_PIXELS = 10    # connected regions smaller than this are treated as noise
CONFIDENCE_THRESHOLD = 0.60  # minimum U-Net segmentation confidence to accept a detection

# ------------------------------------------------------------ severity
SEVERITY_CLASSES = ["Low", "Medium", "High"]
# Heuristic used ONLY by scripts/assign_severity_labels.py --method rule
RULE_AREA_RATIO_LOW_MAX = 0.02      # area_ratio <  0.02  -> Low
RULE_AREA_RATIO_MEDIUM_MAX = 0.08   # area_ratio <  0.08  -> Medium, otherwise High
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

# ------------------------------------------------------------ speed (km/h)
LOW_SPEED = 40
MEDIUM_SPEED = 25
HIGH_SPEED = 10

# ------------------------------------------------------------ live camera / voice
CAMERA_INDEX = 0
INFERENCE_INTERVAL_SEC = 1.0   # run the model at most once per N seconds
VOICE_COOLDOWN_SEC = 10.0      # minimum gap between two spoken alerts
