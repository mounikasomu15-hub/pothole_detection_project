# Architecture
Streamlit (`app.py`) -> `inference/pipeline.py` -> `preprocessing/preprocess_images.py` -> `inference/segmentation.py` (U-Net, `models/unet.py`)
-> `inference/pothole_detector.py` -> `inference/feature_extraction.py` -> `inference/severity_prediction.py` (Random Forest)
-> `utils/speed_recommendation.py` -> `utils/voice_alert.py`. Predictions are appended to `outputs/predictions/predictions.csv`.

* **Single preprocessing code path** for training and inference.
* **Checkpoints are self-describing** (U-Net stores image size and width; the Random Forest stores feature names and label source).
* **Configuration** lives only in `config.py`. `POTHOLE_WORKDIR` relocates data/outputs (used by the smoke test).
* **Failure handling:** missing models -> `ModelNotFoundError` shown as a Streamlit message; unreadable uploads -> friendly error;
  empty segmentation -> "no pothole detected"; missing camera -> message.
