"""
trash_detection.py
--------------------
Litter / on-road-object detection.

IMPORTANT — read before trusting this module's output:
Stock YOLOv8 weights (yolov8n/s/m/l/x.pt) are trained on COCO-80, which has
NO litter/trash classes (no "plastic bag", "paper", "cup lying on road",
etc). There is no honest way to get real trash-type classification out of
a COCO-pretrained model.

What this module actually does, in two modes:

  1. "proxy" mode (default, works out of the box): treats a handful of
     small, loose COCO objects (bottle, cup, etc. — see
     yolov8_detector.TRASH_PROXY_CLASS_IDS) as stand-ins for litter, purely
     so the pipeline and UI are end-to-end functional on the demo footage.
     Labels are shown as "possible litter: <coco class>" — never
     mislabeled as a real trash taxonomy.

  2. "custom" mode: if you point this module at a model fine-tuned on a
     litter dataset (e.g. TACO — http://tacodataset.org/, which has real
     categories like plastic, paper, metal, glass, cigarette), it will use
     that model's own class names directly, and the accuracy numbers
     become meaningful.

The README explains how to fine-tune a custom model in ~1-2 hours on a
free Colab GPU using TACO or a similarly labeled dataset.
"""
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np

from yolov8_detector import YOLOv8Detector, Detection, TRASH_PROXY_CLASS_IDS


@dataclass
class TrashDetectionResult:
    detections: List[Detection]
    mode: str  # "proxy" | "custom"


class TrashDetector:
    def __init__(self, detector: YOLOv8Detector, mode: str = "proxy",
                 custom_weights_path: Optional[str] = None):
        self.mode = mode
        if mode == "custom" and custom_weights_path:
            self.detector = YOLOv8Detector(weights_path=custom_weights_path)
        else:
            self.detector = detector

    def detect(self, frame: np.ndarray, conf: float = 0.30) -> TrashDetectionResult:
        classes = None if self.mode == "custom" else list(TRASH_PROXY_CLASS_IDS)
        dets = self.detector.detect(frame, classes=classes, conf=conf)
        return TrashDetectionResult(detections=dets, mode=self.mode)


def draw_trash(frame: np.ndarray, result: TrashDetectionResult) -> np.ndarray:
    out = frame.copy()
    prefix = "" if result.mode == "custom" else "possible litter: "
    for det in result.detections:
        x1, y1, x2, y2 = map(int, det.box)
        color = (0, 140, 255)  # orange
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}{det.cls_name} {det.conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def counts_by_class(result: TrashDetectionResult) -> dict:
    counts: dict = {}
    for det in result.detections:
        counts[det.cls_name] = counts.get(det.cls_name, 0) + 1
    return counts
