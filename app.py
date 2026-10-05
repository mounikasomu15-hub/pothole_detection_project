"""Streamlit dashboard. Run with:  streamlit run app.py"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

import config
from inference.pipeline import ModelNotFoundError, PotholePipeline
from utils.helpers import append_prediction, read_history, read_json, setup_logging
from utils.speed_recommendation import DISCLAIMER
from utils.visualization import mask_to_rgb
from utils.voice_alert import VoiceAlert

TITLE = "AI-Enabled Vehicle-Mounted System for Real-Time Pothole Detection and Adaptive Speed Recommendation"
st.set_page_config(page_title="Pothole Detection", page_icon="🚧", layout="wide")
log = setup_logging("inference")

PIPELINE_STEPS = ["Road Image / Live Camera", "Preprocessing", "U-Net Segmentation", "Feature Extraction",
                   "Random Forest", "Severity Classification", "Speed Recommendation", "Voice Alert"]
CONF_HELP = ("Segmentation confidence = mean U-Net probability over the pixels it labelled as pothole. "
             "Severity confidence = Random Forest predict_proba() of the winning class (share of trees' "
             "probability mass). Neither is a guarantee of real-world correctness.")


# ------------------------------------------------------------------ models
def _mtime(p):
    return Path(p).stat().st_mtime if Path(p).exists() else 0


@st.cache_resource(show_spinner="Loading models...")
def load_pipeline(unet_mtime, rf_mtime):
    try:
        return PotholePipeline(), None
    except ModelNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # corrupted checkpoint etc.
        return None, f"Could not load the models: {exc}"


def get_pipeline():
    return load_pipeline(_mtime(config.UNET_CKPT), _mtime(config.RF_CKPT))


def get_voice() -> VoiceAlert:
    if "voice" not in st.session_state:
        st.session_state["voice"] = VoiceAlert()
    return st.session_state["voice"]


# ------------------------------------------------------------------ UI helpers
def render_result(image, res):
    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Original image")
    c2.image(mask_to_rgb(res["mask"]), caption="Predicted segmentation mask")
    c3.image(res["overlay"], caption="Overlay + bounding box")
    if not res["detected"]:
        st.success("Pothole: NOT DETECTED. " + res["message"])
        st.metric("Segmentation confidence", f"{res['segmentation_confidence']:.1%}" if res["segmentation_confidence"] else "n/a")
        return
    st.error("Pothole: DETECTED")
    m = st.columns(5)
    m[0].metric("Severity", res["severity"].upper())
    m[1].metric("Severity confidence", f"{res['severity_confidence']:.1%}")
    m[2].metric("Segmentation confidence", f"{res['segmentation_confidence']:.1%}")
    m[3].metric("Pothole area", f"{res['area_px']} px ({res['area_ratio']:.2%})")
    m[4].metric("Recommended safe speed", f"{res['speed']} km/h")
    st.warning(res["explanation"])
    st.caption(DISCLAIMER)
    with st.expander("What do the confidence scores mean? / extracted features"):
        st.write(CONF_HELP)
        st.write("Severity is an **image-based estimate** from the 2-D shape of the segmented region. "
                 "It is **not** a measurement of physical pothole depth.")
        st.json({k: round(float(v), 4) for k, v in res["features"].items()})
        st.write("Random Forest class probabilities:", res["severity_probabilities"])


def settings_sidebar():
    st.sidebar.header("Settings")
    s = {
        "conf": st.sidebar.slider("Confidence threshold", 0.50, 0.99, float(config.CONFIDENCE_THRESHOLD), 0.01,
                                   help="Minimum U-Net segmentation confidence to report a pothole."),
        "seg": st.sidebar.slider("Segmentation threshold", 0.10, 0.90, float(config.SEGMENTATION_THRESHOLD), 0.05,
                                  help="Pixel probability above which a pixel counts as pothole."),
        "voice": st.sidebar.checkbox("Enable Voice Alert", value=False),
        "interval": st.sidebar.slider("Camera processing interval (s)", 0.2, 5.0, float(config.INFERENCE_INTERVAL_SEC), 0.1,
                                       help="The model runs at most once per interval (keeps laptops responsive)."),
        "cam": st.sidebar.number_input("Webcam index", 0, 10, config.CAMERA_INDEX),
    }
    return s


def run_kwargs(s):
    return dict(seg_threshold=s["seg"], conf_threshold=s["conf"])


def show_history():
    hist = read_history(15)
    st.subheader("Recent predictions")
    if hist is None or hist.empty:
        st.caption("No predictions saved yet.")
    else:
        st.dataframe(hist)


# ------------------------------------------------------------------ pages
def page_home():
    st.title(TITLE)
    st.write("A road-safety prototype that segments potholes with a **U-Net**, extracts shape features, "
             "classifies **image-based severity** with a **Random Forest**, and recommends a safe speed with a voice alert.")
    st.subheader("Pipeline")
    st.markdown("\n\n".join(("⬇️ " if i else "") + f"**{step}**" for i, step in enumerate(PIPELINE_STEPS)))
    st.subheader("Objectives")
    st.markdown("- Detect potholes in road images and camera frames\n- Segment the exact pothole region\n"
                "- Estimate severity (Low / Medium / High) from segmentation-derived features\n"
                "- Recommend an adaptive safe speed and warn the driver by voice")
    pipe, err = get_pipeline()
    st.info("Models loaded and ready." if pipe else err)
    st.caption(DISCLAIMER)


def page_image(s):
    st.header("Image Detection")
    pipe, err = get_pipeline()
    if pipe is None:
        st.error(err + "\n\nRun the training steps from the README first.")
        return

    input_mode = st.radio("Choose Input Method", ["Upload Custom Image", "Select from Full Dataset (Train/Val/Test)"], horizontal=True)
    
    image = None
    file_name = None
    file_size = 0

    if input_mode == "Upload Custom Image":
        up = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
        if up is not None:
            file_name = up.name
            file_size = up.size
            try:
                image = np.array(ImageOps.exif_transpose(Image.open(up)).convert("RGB"), dtype=np.uint8)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be read as an image (corrupted or unsupported format).")
                return
    else:
        dataset_dir = Path("dataset")
        if dataset_dir.exists():
            # Scan all subfolders (train, val, test, etc.) for images excluding masks
            sample_images = sorted([
                str(f.relative_to(dataset_dir)) for f in dataset_dir.rglob("*") 
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg') and "mask" not in f.parts
            ])
            if sample_images:
                selected_sample = st.selectbox("Choose an image from dataset:", sample_images)
                image_path = dataset_dir / selected_sample
                file_name = Path(selected_sample).name
                file_size = image_path.stat().st_size
                try:
                    image = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"), dtype=np.uint8)
                except (UnidentifiedImageError, OSError, ValueError):
                    st.error("This sample file could not be read as an image.")
                    return
            else:
                st.warning("No images found in dataset directory.")
        else:
            st.warning("Dataset directory 'dataset' not found.")

    if image is None:
        show_history()
        return

    with st.spinner("Running pipeline..."):
        res = pipe.run(image, **run_kwargs(s))
    render_result(image, res)

    key = f"{file_name}-{file_size}"
    if st.session_state.get("logged_file") != key:         # log + auto-speak once per selection
        append_prediction(file_name, res)
        st.session_state["logged_file"] = key
        if s["voice"] and res["detected"]:
            get_voice().alert(res["severity"], res["speed"], force=True)
    if res["detected"] and st.button("🔊 Play voice alert"):
        get_voice().alert(res["severity"], res["speed"], force=True)
    show_history()


def process_frame(pipe, rgb, s, source_name):
    res = pipe.run(rgb, **run_kwargs(s))
    if res["detected"]:
        voice = get_voice()
        if voice.ready():
            append_prediction(source_name, res)
            if s["voice"]:
                voice.alert(res["severity"], res["speed"])
    return res


def status_text(res):
    if not res["detected"]:
        return "✅ No pothole detected"
    return (f"🚧 **Pothole detected** | Severity **{res['severity']}** | severity confidence "
            f"{res['severity_confidence']:.0%} | segmentation confidence {res['segmentation_confidence']:.0%} "
            f"| Recommended speed **{res['speed']} km/h**")


def page_live(s):
    st.header("Live Camera")
    pipe, err = get_pipeline()
    if pipe is None:
        st.error(err + "\n\nRun the training steps from the README first."); return
    st.caption("Streamlit cannot stream browser video continuously through Python, so two practical modes are offered.")
    tab1, tab2 = st.tabs(["Browser camera (snapshot)", "Continuous webcam (local OpenCV)"])
    with tab1:
        st.write("Works in any browser and on phones. Take a photo; it is processed instantly.")
        shot = st.camera_input("Take a picture of the road")
        if shot is not None:
            rgb = np.array(Image.open(shot).convert("RGB"), dtype=np.uint8)
            res = process_frame(pipe, rgb, s, "browser_camera")
            render_result(rgb, res)
    with tab2:
        st.write("Reads the webcam of the computer **running Streamlit** (works when you run it locally). "
                 f"The model runs every {s['interval']:.1f}s to keep the laptop responsive. "
                 "Turn the switch off to stop.")
        if st.toggle("Start continuous webcam"):
            cap = cv2.VideoCapture(int(s["cam"]), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(s["cam"]))
            if not cap.isOpened():
                cap.release()
                st.error("Camera unavailable. Check that it is connected, not used by another app, "
                         "and try another webcam index in the sidebar.")
                return
            left, right = st.columns(2)
            raw_ph, res_ph, info_ph = left.empty(), right.empty(), st.empty()
            last = 0.0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        st.error("Could not read a frame from the camera."); break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_ph.image(rgb, caption="Camera frame")
                    if time.time() - last >= s["interval"]:
                        last = time.time()
                        res = process_frame(pipe, rgb, s, "live_camera")
                        res_ph.image(res["overlay"], caption="Segmentation overlay (latest processed frame)")
                        info_ph.markdown(status_text(res))
                    time.sleep(0.03)
            finally:
                cap.release()


def metric_block(title, data, keys):
    st.subheader(title)
    cols = st.columns(len(keys))
    for c, (k, label) in zip(cols, keys):
        c.metric(label, f"{data[k]:.4f}")


def page_performance():
    st.header("Model Performance")
    st.caption("All values below are loaded from files written by the evaluation scripts - nothing is hard-coded.")
    um = read_json(config.REPORTS_DIR / "unet_metrics.json")
    hist = read_json(config.REPORTS_DIR / "unet_history.json")
    rf = read_json(config.REPORTS_DIR / "rf_metrics.json")

    if um:
        metric_block(f"U-Net ({um.get('split', 'test')} set, {um.get('num_images')} images)", um,
                     [("iou", "IoU"), ("dice", "Dice"), ("precision", "Precision"), ("recall", "Recall"),
                      ("f1", "F1-score"), ("pixel_accuracy", "Pixel accuracy")])
    else:
        st.subheader("U-Net"); st.warning("Model not trained. Run training first.  (python training/train_unet.py, then evaluate_unet.py)")
    if hist:
        df = pd.DataFrame({"Training loss": hist["train_loss"], "Validation loss": hist["val_loss"]})
        df.index = range(1, len(df) + 1)
        a, b = st.columns(2)
        a.write("**Training loss**"); a.line_chart(df[["Training loss"]])
        b.write("**Validation loss**"); b.line_chart(df[["Validation loss"]])
    samples = config.GRAPHS_DIR / "unet_samples.png"
    if samples.exists():
        st.image(str(samples), caption="Sample predictions (test set)")

    if rf:
        metric_block(f"Random Forest ({rf['num_test_samples']} test samples)", rf,
                     [("accuracy", "Accuracy"), ("precision_weighted", "Precision (weighted)"),
                      ("recall_weighted", "Recall (weighted)"), ("f1_weighted", "F1 (weighted)")])
        srcs = ", ".join(rf.get("label_sources") or ["unknown"])
        st.info(f"Severity label source: **{srcs}**. Rule-based labels are derived from the same features the "
                "forest uses, so high accuracy mostly shows it re-learned the rule.")
        cm = config.GRAPHS_DIR / "rf_confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), caption="Confusion matrix (test set)")
    else:
        st.subheader("Random Forest"); st.warning("Model not trained. Run training first.  (python training/train_random_forest.py, then evaluate_random_forest.py)")


def page_about():
    st.header("About Project")
    st.write(TITLE)
    st.markdown(
        "**Models:** U-Net (PyTorch, from scratch) for segmentation; Random Forest (scikit-learn) for severity.\n\n"
        "**Severity** is an *image-based estimate* from segmented shape features (area ratio, size, shape). "
        "A normal RGB camera cannot measure physical depth, and this system does not claim to.\n\n"
        "**Speed recommendations** are configurable in `config.py` and are for demonstration only - not certified road-safety limits.\n\n"
        "**Limitations:** results depend on the training data and lighting; the Random Forest is only as good as its severity labels; "
        "continuous browser video is limited in Streamlit.")
    st.caption(DISCLAIMER)


def main():
    st.sidebar.title("🚧 Pothole Detection")
    page = st.sidebar.radio("Navigation", ["Home", "Image Detection", "Live Camera", "Model Performance", "About Project"])
    s = settings_sidebar()
    {"Home": page_home, "Image Detection": lambda: page_image(s), "Live Camera": lambda: page_live(s),
     "Model Performance": page_performance, "About Project": page_about}[page]()


main()