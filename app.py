"""
app.py
------
Streamlit UI for the Computer Vision Suite. Four tabs, one per module.
Run with:  streamlit run app.py
"""
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from yolov8_detector import YOLOv8Detector, VEHICLE_CLASS_IDS
from parking_spot import (
    load_slots_from_list, update_occupancy, occupancy_stats,
    draw_slots, default_grid_slots,
)
from speed_estimation import SpeedEstimator, SpeedEstimatorConfig, draw_speeds
from wrong_way_detection import WrongWayDetector, WrongWayConfig, draw_wrong_way
from trash_detection import TrashDetector, draw_trash, counts_by_class

MODELS_DIR = Path("models")
DEFAULT_WEIGHTS = str(MODELS_DIR / "yolov8n.pt")
TARGET_FPS = 12  # within the 10-15 FPS spec; frames are throttled to this in the UI loop

st.set_page_config(page_title="Computer Vision Suite - YOLOv8", layout="wide")


def load_css():
    css_path = Path("static/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_detector(weights_path: str) -> YOLOv8Detector:
    return YOLOv8Detector(weights_path=weights_path)


def read_video_frames(video_path: str, max_frames: int = None):
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_skip = max(1, round(src_fps / TARGET_FPS))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % frame_skip == 0:
            yield frame
        i += 1
        if max_frames and i >= max_frames:
            break
    cap.release()


def sidebar_model_config():
    st.sidebar.header("Model")
    weights_path = st.sidebar.text_input("YOLOv8 weights path", value=DEFAULT_WEIGHTS)
    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    if not Path(weights_path).exists():
        st.sidebar.warning(
            "Weights file not found locally. Ultralytics will attempt to "
            "auto-download standard weights (e.g. yolov8n.pt) on first "
            "run — requires internet access."
        )
    return weights_path, conf


# ----------------------------------------------------------------------
# Tab 1: Parking Spot Finder
# ----------------------------------------------------------------------
def tab_parking(weights_path, conf):
    st.subheader("Parking Spot Finder")
    st.caption(
        "Vehicles are detected each frame and checked for overlap against "
        "fixed parking-slot boxes. Slots are green (free) or red (occupied)."
    )
    source = st.file_uploader("Upload a parking-lot image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    rows = st.sidebar.number_input("Grid rows (demo layout)", 1, 6, 2)
    cols = st.sidebar.number_input("Grid cols (demo layout)", 1, 10, 5)
    overlap_thresh = st.sidebar.slider("Occupancy overlap threshold", 0.1, 0.9, 0.30, 0.05)

    if not source:
        st.info("Upload an image/video, or check `test_assets/` for a generated sample.")
        return

    detector = get_detector(weights_path)
    is_image = source.type.startswith("image")

    if is_image:
        file_bytes = np.frombuffer(source.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w = frame.shape[:2]
        slots = load_slots_from_list(default_grid_slots(w, h, rows, cols))
        dets = detector.detect(frame, classes=list(VEHICLE_CLASS_IDS), conf=conf)
        slots = update_occupancy(slots, dets, overlap_thresh)
        annotated = draw_slots(frame, slots)
        stats = occupancy_stats(slots)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        with col2:
            st.metric("Occupancy", f"{stats['pct_occupied']}%")
            st.metric("Occupied", stats["occupied"])
            st.metric("Free", stats["free"])
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(source.read())
        frame_slot = st.empty()
        metric_slot = st.empty()
        slots = None
        for frame in read_video_frames(tmp.name, max_frames=300):
            if slots is None:
                h, w = frame.shape[:2]
                slots = load_slots_from_list(default_grid_slots(w, h, rows, cols))
            dets = detector.detect(frame, classes=list(VEHICLE_CLASS_IDS), conf=conf)
            slots = update_occupancy(slots, dets, overlap_thresh)
            annotated = draw_slots(frame, slots)
            stats = occupancy_stats(slots)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            metric_slot.markdown(
                f"**Occupancy: {stats['pct_occupied']}%**  "
                f"({stats['occupied']} occupied / {stats['free']} free of {stats['total']})"
            )
            time.sleep(1.0 / TARGET_FPS)


# ----------------------------------------------------------------------
# Tab 2: Speed Estimation
# ----------------------------------------------------------------------
def tab_speed(weights_path, conf):
    st.subheader("Speed Estimation")
    st.caption(
        "Vehicles are tracked frame-to-frame with ByteTrack; pixel "
        "displacement is converted to km/h via a configurable scale."
    )
    source = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"], key="speed_upload")
    scale = st.sidebar.number_input(
        "Meters per pixel (calibration)", min_value=0.001, max_value=2.0,
        value=0.05, step=0.005, format="%.3f",
        help="Measure a known real-world distance (e.g. lane width) in the "
             "video, divide by its pixel length, enter that here.",
    )
    speed_limit = st.sidebar.number_input("Speed limit for alert (km/h, 0=off)", 0, 300, 0)

    if not source:
        st.info("Upload a traffic video to run speed estimation.")
        return

    detector = get_detector(weights_path)
    estimator = SpeedEstimator(SpeedEstimatorConfig(meters_per_pixel=scale))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(source.read())
    frame_slot = st.empty()
    table_slot = st.empty()

    for frame in read_video_frames(tmp.name, max_frames=300):
        dets = detector.track(frame, classes=list(VEHICLE_CLASS_IDS), conf=conf)
        speeds = estimator.update(dets, frame_time=time.time())
        limit = speed_limit if speed_limit > 0 else None
        annotated = draw_speeds(frame, dets, speeds, speed_limit_kmh=limit)
        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if speeds:
            rows = [{"track_id": k, "speed_kmh": round(v, 1)} for k, v in speeds.items()]
            table_slot.dataframe(rows, use_container_width=True, hide_index=True)
        time.sleep(1.0 / TARGET_FPS)


# ----------------------------------------------------------------------
# Tab 3: Wrong-Way Detection
# ----------------------------------------------------------------------
def tab_wrong_way(weights_path, conf):
    st.subheader("Wrong-Way Detection")
    st.caption(
        "Define the allowed direction of travel; vehicles whose tracked "
        "movement deviates sharply from it are flagged."
    )
    source = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"], key="wrongway_upload")
    direction = st.sidebar.selectbox(
        "Allowed direction", ["Down (top->bottom)", "Up (bottom->top)", "Left (right->left)", "Right (left->right)"],
    )
    angle_thresh = st.sidebar.slider("Flag angle threshold (deg)", 60, 170, 110)
    dir_map = {
        "Down (top->bottom)": (0.0, 1.0),
        "Up (bottom->top)": (0.0, -1.0),
        "Left (right->left)": (-1.0, 0.0),
        "Right (left->right)": (1.0, 0.0),
    }

    if not source:
        st.info("Upload a traffic video to run wrong-way detection.")
        return

    detector = get_detector(weights_path)
    cfg = WrongWayConfig(allowed_direction=dir_map[direction], angle_threshold_deg=angle_thresh)
    wwd = WrongWayDetector(cfg)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(source.read())
    frame_slot = st.empty()
    alert_slot = st.empty()
    total_alerts = set()

    for frame in read_video_frames(tmp.name, max_frames=300):
        dets = detector.track(frame, classes=list(VEHICLE_CLASS_IDS), conf=conf)
        flags = wwd.update(dets)
        annotated = draw_wrong_way(frame, dets, flags, cfg.allowed_direction)
        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        for tid, flagged in flags.items():
            if flagged:
                total_alerts.add(tid)
        if total_alerts:
            alert_slot.error(f"Wrong-way vehicles flagged so far: {sorted(total_alerts)}")
        time.sleep(1.0 / TARGET_FPS)


# ----------------------------------------------------------------------
# Tab 4: Trash Detection
# ----------------------------------------------------------------------
def tab_trash(weights_path, conf):
    st.subheader("Trash Detection")
    st.warning(
        "**Accuracy note:** stock YOLOv8/COCO weights have no litter "
        "classes. This tab runs in *proxy mode* by default (loose objects "
        "like bottles/cups stand in for litter) purely to demo the "
        "pipeline. Point it at a custom-trained model (see README) for "
        "real trash-type classification.",
        icon="⚠️",
    )
    mode = st.sidebar.radio("Trash detection mode", ["proxy (demo)", "custom model"])
    custom_path = None
    if mode == "custom model":
        custom_path = st.sidebar.text_input("Path to custom .pt weights", value="models/trash_custom.pt")

    source = st.file_uploader("Upload road/surface image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], key="trash_upload")
    if not source:
        st.info("Upload an image/video to run trash detection.")
        return

    base_detector = get_detector(weights_path)
    trash_detector = TrashDetector(
        base_detector,
        mode="custom" if mode == "custom model" else "proxy",
        custom_weights_path=custom_path,
    )
    is_image = source.type.startswith("image")

    if is_image:
        file_bytes = np.frombuffer(source.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = trash_detector.detect(frame, conf=conf)
        annotated = draw_trash(frame, result)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        with col2:
            st.write("**Detections by class**")
            st.json(counts_by_class(result))
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(source.read())
        frame_slot = st.empty()
        stat_slot = st.empty()
        for frame in read_video_frames(tmp.name, max_frames=300):
            result = trash_detector.detect(frame, conf=conf)
            annotated = draw_trash(frame, result)
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            stat_slot.json(counts_by_class(result))
            time.sleep(1.0 / TARGET_FPS)


def main():
    load_css()
    st.title("🚗 Computer Vision Suite — YOLOv8")
    st.caption("Parking · Speed · Wrong-Way · Trash — all built on one shared YOLOv8 detector/tracker.")

    weights_path, conf = sidebar_model_config()

    tabs = st.tabs(["🅿️ Parking Spot Finder", "⚡ Speed Estimation", "🚫 Wrong-Way Detection", "🗑️ Trash Detection"])
    with tabs[0]:
        tab_parking(weights_path, conf)
    with tabs[1]:
        tab_speed(weights_path, conf)
    with tabs[2]:
        tab_wrong_way(weights_path, conf)
    with tabs[3]:
        tab_trash(weights_path, conf)


if __name__ == "__main__":
    main()
