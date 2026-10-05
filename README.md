# AI-Enabled Vehicle-Mounted System for Real-Time Pothole Detection and Adaptive Speed Recommendation

> **Results will be generated after training.** This README contains no accuracy, IoU, Dice or F1 numbers
> because none exist until you train on your own dataset. The Streamlit "Model Performance" page loads the
> metrics that your own evaluation scripts write to `outputs/reports/`.

## 1. Project overview
Road image / camera frame -> **U-Net** pothole segmentation -> shape features -> **Random Forest** severity
(Low / Medium / High) -> safe-speed recommendation -> offline voice alert, shown in a **Streamlit** dashboard.

**Honesty notes**
* **Severity is image-based.** It is estimated from the 2-D segmented shape/size. A normal RGB camera does *not* measure physical depth.
* **Severity labels:** public pothole datasets normally have masks but *no* severity labels. This project provides
  (a) a **rule-based** labelling script (documented thresholds in `config.py`) or (b) a CSV you label **manually**.
  Rule-based labels are derived from the same features the forest sees, so the forest largely re-learns the rule.
  Report the label source in your thesis (the app and evaluation script print it).
* **Speeds** (40 / 25 / 10 km/h) are configurable demo values, **not certified road-safety limits**.
* **Dataset:** no real images are bundled. You must supply ~5000 real images + masks.

## 2. Architecture
```
Image/Camera -> Preprocess (resize, normalise) -> U-Net -> probability map -> threshold -> mask
   -> detect (area >= MIN_AREA_RATIO and confidence >= threshold)
   -> features (area, bbox, perimeter, circularity, ...) -> Random Forest -> severity + confidence
   -> speed (config.py) -> voice alert (cooldown) -> Streamlit
```
The same `preprocessing/preprocess_images.py` is used for training and inference. The U-Net checkpoint stores
`image_size` and `base_channels`, so inference always matches training.

## 3. Folder structure
```
app.py  config.py  requirements.txt  README.md
dataset/{images,masks,train,val,test}   severity_labels.csv (generated)
models/        unet.py  random_forest.py
preprocessing/ preprocess_images.py  dataset.py  prepare_dataset.py  split_dataset.py
training/      train_unet.py  evaluate_unet.py  train_random_forest.py  evaluate_random_forest.py
inference/     pipeline.py  segmentation.py  pothole_detector.py  feature_extraction.py  severity_prediction.py
utils/         helpers.py  metrics.py  visualization.py  speed_recommendation.py  voice_alert.py
scripts/       validate_dataset.py  generate_feature_dataset.py  assign_severity_labels.py  run_pipeline.py  smoke_test.py
checkpoints/ (unet_best.pth, random_forest.pkl appear after training)   outputs/   logs/   docs/
```

## 4-7. Requirements, Python, virtual environment, installation (Windows)
Python **3.10 - 3.12 (64-bit)**. Open a terminal in the project folder:
```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/smoke_test.py
```
`smoke_test.py` uses *synthetic* data in a temp folder to confirm that everything runs (results are meaningless and discarded).
For an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org), otherwise the CPU build is used automatically.
The program prints `Using device: CPU` or `Using device: CUDA`.

## 8-9. Dataset preparation and structure
Collect **~5000 pothole images with segmentation masks** (e.g. from Kaggle/Roboflow/academic pothole-segmentation sets - check each licence).
Final layout:
```
dataset/images/image_0001.jpg ... image_5000.jpg
dataset/masks/image_0001.png  ... image_5000.png     (white/255 = pothole, black/0 = background)
```
`image_0001.jpg` and `image_0001.png` are matched by file name. If your files are elsewhere/named differently:
```
python preprocessing/prepare_dataset.py --images "C:\path\to\images" --masks "C:\path\to\masks" --rename
```
This copies pairs into `dataset/`, converts masks to clean binary PNG and optionally renames them.
(Images without potholes are allowed - use an all-black mask.)

## 10. Dataset validation
```
python scripts/validate_dataset.py
```
Checks missing masks, orphan masks, duplicate names, corrupted files, size mismatches, invalid mask values, and prints statistics.

## 11. Dataset splitting (70 / 15 / 15)
```
python preprocessing/split_dataset.py
```
Copies files into `dataset/train|val|test`. Re-split with `--clean`. Seed = `RANDOM_SEED` in `config.py`.

## 12. U-Net training
```
python training/train_unet.py
```
Options: `--epochs 40 --batch-size 16 --lr 0.001`. Loss = BCE + Dice, augmentation = flip / +-10 deg rotation /
brightness-contrast (mask transformed identically; nearest-neighbour). Best model (by validation Dice) ->
`checkpoints/unet_best.pth`; curves -> `outputs/graphs/unet_training.png`; log -> `logs/train_unet.log`.
CPU tip: lower `UNET_BASE_CHANNELS` to 16 or `BATCH_SIZE` to 8 in `config.py`.

## 13. U-Net evaluation
```
python training/evaluate_unet.py
```
Computes IoU, Dice, precision, recall, F1, pixel accuracy on the **test** split -> `outputs/reports/unet_metrics.json`,
and saves original / ground truth / prediction / overlay examples -> `outputs/graphs/unet_samples.png`.

## 14. Feature extraction and severity labels
```
python scripts/generate_feature_dataset.py            # writes dataset/severity_labels.csv (no severity yet)
python scripts/assign_severity_labels.py --method rule        # rule-based (config.py thresholds)
python scripts/assign_severity_labels.py --method percentile  # rule-based (terciles of your data)
python scripts/assign_severity_labels.py --method check       # validate labels you typed manually
```
**Better:** open `dataset/severity_labels.csv` and type `Low`/`Medium`/`High` into the `severity` column yourself
(set `severity_source` to e.g. `manual`). Existing labels are preserved when you regenerate features.
Optional: `generate_feature_dataset.py --source pred` extracts features from *U-Net predicted* masks (closer to what the forest sees at inference).

## 15-16. Random Forest training and evaluation
```
python training/train_random_forest.py
python training/evaluate_random_forest.py
```
Outputs: `checkpoints/random_forest.pkl`, `outputs/reports/rf_metrics.json` (accuracy, precision, recall, F1, per-class report),
`outputs/graphs/rf_confusion_matrix.png`. Uses the same train/val/test membership as the U-Net when the split folders exist.

## 17. Running Streamlit
```
streamlit run app.py
```
If the models are not trained you get "Model files not found. Please train the models first." instead of a crash.

## 18. Image detection
Sidebar -> **Image Detection** -> upload JPG/PNG. Shows original, mask, overlay with bounding box, status, severity,
segmentation confidence, severity confidence, area, recommended speed, and a *Play voice alert* button. Predictions are logged to
`outputs/predictions/predictions.csv` and recent ones are listed.
*Confidence:* segmentation confidence = mean U-Net probability over predicted pothole pixels; severity confidence = Random Forest `predict_proba()` of the chosen class.

## 19. Live camera
* **Browser camera (snapshot)** - works everywhere, including phones.
* **Continuous webcam** - OpenCV reads the camera of the machine running Streamlit; the model runs every *N* seconds (sidebar slider).

*Limitation:* Streamlit cannot stream browser video frames to Python continuously without extra components, so continuous mode needs a local webcam.

## 20. Voice alert
Tick **Enable Voice Alert** in the sidebar. Offline `pyttsx3` speaks e.g. "Warning. Medium severity pothole detected. Reduce speed to 25 kilometers per hour."
A cooldown (`VOICE_COOLDOWN_SEC`, default 10 s) prevents repeated alerts. Audio plays on the computer running Streamlit.

## 21. Troubleshooting
| Problem | Fix |
|---|---|
| `No image/mask pairs found` | Put files in `dataset/images` and `dataset/masks`; names must match (`image_0001.jpg` / `image_0001.png`). |
| `split folders already contain files` | `python preprocessing/split_dataset.py --clean` |
| `Model files not found` | Train: U-Net, then features -> labels -> Random Forest. |
| `No labelled rows found` | Run `scripts/assign_severity_labels.py` or fill the `severity` column. |
| Only one severity class | Adjust `RULE_AREA_RATIO_*` in `config.py` or use `--method percentile`. |
| Out of memory / slow | Lower `BATCH_SIZE`, `UNET_BASE_CHANNELS`, or `IMAGE_SIZE` (then retrain everything). |
| DataLoader errors on Windows | Keep `NUM_WORKERS = 0`. |
| No voice | `pip install pyttsx3`; check Windows speakers/voices. |
| Camera unavailable | Close other apps using it; change webcam index in the sidebar. |
| `pip` cannot find a torch version | Use Python 3.10-3.12 64-bit. |

## 22. Limitations
Depends on real data quality; image-based severity (no physical depth, no camera calibration); rule-based labels unless you annotate;
predicted-mask features differ slightly from ground-truth-mask features; changing `IMAGE_SIZE` requires retraining both models;
speeds are demonstration values; not a certified safety system.

## 23. Future enhancements
Depth/stereo/LiDAR sensors for real depth; manual expert severity labels; GPS + speed-sensor integration; video tracking across frames;
ONNX / mobile deployment; larger backbones (e.g. U-Net with pretrained encoder).
