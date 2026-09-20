"""
yolov8_detector.py
-------------------
Thin, shared wrapper around the Ultralytics YOLOv8 model.

Every module in this suite (parking, speed, wrong-way, trash) goes through
this class instead of touching `ultralytics.YOLO` directly. That gives us:
  - one place to swap model weights (e.g. yolov8n.pt -> a custom trash model)
  - one place to configure tracking (ByteTrack via `model.track`)
  - a consistent Detection data format for the rest of the app

COCO class ids used (standard yolov8*.pt weights):
  0  person
  2  car
  3  motorcycle
  5  bus
  7  truck
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Stand-in classes used to *simulate* litter detection with stock COCO
# weights. THIS IS A PLACEHOLDER — see README "Known Limitations". A real
# trash detector needs to be fine-tuned on a litter dataset (e.g. TACO).
TRASH_PROXY_CLASS_IDS = {39, 41, 44, 67}  # bottle, cup, spoon, cell phone


@dataclass
class Detection:
    box: tuple            # (x1, y1, x2, y2) in pixels
    conf: float
    cls_id: int
    cls_name: str
    track_id: Optional[int] = None
    centroid: tuple = field(default=(0, 0))

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        self.centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class YOLOv8Detector:
    """
    Lazily loads a YOLOv8 model on first use so importing this module
    (e.g. for unit tests) never triggers a network call or GPU init.
    """

    def __init__(self, weights_path: str = "models/yolov8n.pt", device: Optional[str] = None):
        self.weights_path = weights_path
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
        return self._model

    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None,
               conf: float = 0.35) -> List[Detection]:
        """Single-frame detection, no tracking (used by Parking + Trash)."""
        results = self.model.predict(
            source=frame, conf=conf, classes=classes, verbose=False, device=self.device
        )
        return self._to_detections(results)

    def track(self, frame: np.ndarray, classes: Optional[List[int]] = None,
              conf: float = 0.35, persist: bool = True) -> List[Detection]:
        """Multi-frame tracking (used by Speed + Wrong-Way). Requires a
        stream of frames from the *same* video/stream for track_id continuity."""
        results = self.model.track(
            source=frame, conf=conf, classes=classes, persist=persist,
            tracker="bytetrack.yaml", verbose=False, device=self.device,
        )
        return self._to_detections(results, want_ids=True)

    def reset_tracker(self):
        """Call between separate videos so old track IDs don't bleed over."""
        self._model = None

    def _to_detections(self, results, want_ids: bool = False) -> List[Detection]:
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        names: Dict[int, str] = r.names
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        ids = None
        if want_ids and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            det = Detection(
                box=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(confs[i]),
                cls_id=int(cls_ids[i]),
                cls_name=names.get(int(cls_ids[i]), str(cls_ids[i])),
                track_id=int(ids[i]) if ids is not None else None,
            )
            out.append(det)
        return out
