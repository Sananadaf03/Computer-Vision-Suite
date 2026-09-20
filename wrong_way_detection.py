"""
wrong_way_detection.py
------------------------
Flags vehicles moving opposite to a user-defined "correct" travel direction.

Method:
  1. User (or default config) specifies an allowed-direction vector for a
     lane/road, e.g. "traffic should move downward" -> (0, 1).
  2. For each tracked vehicle we compute its recent movement vector from
     centroid history (same track history mechanism as speed estimation).
  3. We take the cosine similarity (normalized dot product) between the
     vehicle's movement vector and the allowed direction. Below a
     threshold (default: moving more than ~100-120 degrees off-axis)
     the vehicle is flagged wrong-way.

A minimum displacement is required before judging direction, so a car
that's essentially stationary at a light isn't flagged from noise.
"""
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
import math
import cv2
import numpy as np

from yolov8_detector import Detection, VEHICLE_CLASS_IDS

Point = Tuple[float, float]


@dataclass
class WrongWayConfig:
    allowed_direction: Point = (0.0, 1.0)   # unit-ish vector; (0,1) = "should move down the frame"
    history_len: int = 8
    min_displacement_px: float = 12.0       # ignore near-stationary tracks
    angle_threshold_deg: float = 110.0      # flag if movement angle vs allowed > this


class WrongWayDetector:
    def __init__(self, config: WrongWayConfig = None):
        self.cfg = config or WrongWayConfig()
        self._history: Dict[int, Deque[Point]] = defaultdict(
            lambda: deque(maxlen=self.cfg.history_len)
        )

    def reset(self):
        self._history.clear()

    def set_direction(self, dx: float, dy: float):
        norm = math.hypot(dx, dy) or 1.0
        self.cfg.allowed_direction = (dx / norm, dy / norm)

    def update(self, detections: List[Detection]) -> Dict[int, bool]:
        """Returns {track_id: is_wrong_way} for vehicles with enough history."""
        flags: Dict[int, bool] = {}
        ax, ay = self.cfg.allowed_direction
        for det in detections:
            if det.cls_id not in VEHICLE_CLASS_IDS or det.track_id is None:
                continue
            hist = self._history[det.track_id]
            hist.append(det.centroid)
            if len(hist) < 2:
                continue
            (x0, y0), (x1, y1) = hist[0], hist[-1]
            dx, dy = x1 - x0, y1 - y0
            disp = math.hypot(dx, dy)
            if disp < self.cfg.min_displacement_px:
                continue
            # cosine similarity between movement vector and allowed direction
            cos_theta = (dx * ax + dy * ay) / disp
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle_deg = math.degrees(math.acos(cos_theta))
            flags[det.track_id] = angle_deg > self.cfg.angle_threshold_deg
        return flags


def draw_wrong_way(frame: np.ndarray, detections: List[Detection],
                    flags: Dict[int, bool], allowed_direction: Point) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    # draw the reference direction arrow in the corner
    ax, ay = allowed_direction
    cx, cy = 50, 50
    cv2.arrowedLine(out, (cx, cy), (int(cx + ax * 40), int(cy + ay * 40)),
                     (255, 255, 0), 2, tipLength=0.4)
    cv2.putText(out, "allowed dir", (cx - 20, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    for det in detections:
        if det.track_id is None or det.track_id not in flags:
            continue
        x1, y1, x2, y2 = map(int, det.box)
        if flags[det.track_id]:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(out, "WRONG WAY", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 1)
    return out
