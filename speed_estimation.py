"""
speed_estimation.py
--------------------
Real-time vehicle speed estimation from YOLOv8 track history.

Method:
  1. YOLOv8's built-in tracker (ByteTrack) gives each vehicle a persistent
     track_id across frames.
  2. We keep a short rolling history of (frame_time, centroid_px) per
     track_id.
  3. Pixel displacement between two history points is converted to
     real-world distance using a configurable scale factor
     (meters-per-pixel), then divided by elapsed time -> speed.

Why meters-per-pixel and not a homography: a full perspective transform
(bird's-eye calibration) is the "correct" way to do this on angled CCTV
footage, and is offered as the documented upgrade path in the README.
A flat scale factor is the pragmatic default because it needs zero setup
beyond "how many meters does N pixels correspond to in this shot", which
is the ask in this project's spec (configurable pixel-to-real scale).

Speed is smoothed with a short moving average per track to reduce jitter
from per-frame detection noise.
"""
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
import time
import cv2
import numpy as np

from yolov8_detector import Detection, VEHICLE_CLASS_IDS

HistoryPoint = Tuple[float, Tuple[float, float]]  # (timestamp, (cx, cy))


@dataclass
class SpeedEstimatorConfig:
    meters_per_pixel: float = 0.05   # calibrate per-camera; see README
    history_len: int = 10            # frames of centroid history kept per track
    smoothing_window: int = 5        # speeds averaged over last N estimates
    min_frames_for_estimate: int = 3


class SpeedEstimator:
    def __init__(self, config: SpeedEstimatorConfig = None):
        self.cfg = config or SpeedEstimatorConfig()
        self._history: Dict[int, Deque[HistoryPoint]] = defaultdict(
            lambda: deque(maxlen=self.cfg.history_len)
        )
        self._speed_history: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.cfg.smoothing_window)
        )

    def reset(self):
        self._history.clear()
        self._speed_history.clear()

    def update(self, detections: List[Detection], frame_time: float = None) -> Dict[int, float]:
        """Feed one frame's tracked detections; returns {track_id: speed_kmh}."""
        t = frame_time if frame_time is not None else time.time()
        speeds: Dict[int, float] = {}
        for det in detections:
            if det.cls_id not in VEHICLE_CLASS_IDS or det.track_id is None:
                continue
            hist = self._history[det.track_id]
            hist.append((t, det.centroid))
            if len(hist) < self.cfg.min_frames_for_estimate:
                continue
            (t0, p0), (t1, p1) = hist[0], hist[-1]
            dt = t1 - t0
            if dt <= 0:
                continue
            dist_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            dist_m = dist_px * self.cfg.meters_per_pixel
            speed_mps = dist_m / dt
            speed_kmh = speed_mps * 3.6
            sh = self._speed_history[det.track_id]
            sh.append(speed_kmh)
            speeds[det.track_id] = float(np.mean(sh))
        return speeds


def draw_speeds(frame: np.ndarray, detections: List[Detection],
                 speeds: Dict[int, float], speed_limit_kmh: float = None) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        if det.track_id is None or det.track_id not in speeds:
            continue
        x1, y1, x2, y2 = map(int, det.box)
        spd = speeds[det.track_id]
        over_limit = speed_limit_kmh is not None and spd > speed_limit_kmh
        color = (0, 0, 255) if over_limit else (255, 180, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {det.track_id}: {spd:.1f} km/h" + (" !" if over_limit else "")
        cv2.putText(out, label, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out
