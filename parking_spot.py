"""
parking_spot.py
----------------
Parking-spot occupancy detection.

Approach: rather than trying to have YOLO *find* parking-space boundaries
(there's no COCO class for "empty asphalt rectangle"), we take a small,
user-defined list of parking-slot rectangles (drawn once per camera/lot,
since spots don't move) and check each frame's vehicle detections for
overlap against each slot. A slot is "occupied" if any vehicle box overlaps
it above an IoU/overlap threshold, else "empty".

This mirrors how most real parking-occupancy systems actually work
(e.g. PKLot-style datasets): fixed slot polygons + a generic vehicle
detector, not a learned "spot" class.
"""
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

from yolov8_detector import Detection, VEHICLE_CLASS_IDS

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class ParkingSlot:
    id: int
    box: Box
    occupied: bool = False


def load_slots_from_list(boxes: List[Box]) -> List[ParkingSlot]:
    return [ParkingSlot(id=i, box=b) for i, b in enumerate(boxes)]


def _overlap_ratio(slot_box: Box, veh_box: Box) -> float:
    """Fraction of the SLOT's area covered by the vehicle box.
    Using slot-area (not IoU) is more robust here because vehicles are
    often larger than a tight parking-slot box."""
    sx1, sy1, sx2, sy2 = slot_box
    vx1, vy1, vx2, vy2 = veh_box
    ix1, iy1 = max(sx1, vx1), max(sy1, vy1)
    ix2, iy2 = min(sx2, vx2), min(sy2, vy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    slot_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    return inter / slot_area


def update_occupancy(slots: List[ParkingSlot], detections: List[Detection],
                      overlap_thresh: float = 0.30) -> List[ParkingSlot]:
    vehicle_boxes = [d.box for d in detections if d.cls_id in VEHICLE_CLASS_IDS]
    for slot in slots:
        slot.occupied = any(
            _overlap_ratio(slot.box, tuple(map(int, vb))) >= overlap_thresh
            for vb in vehicle_boxes
        )
    return slots


def occupancy_stats(slots: List[ParkingSlot]) -> dict:
    total = len(slots)
    occupied = sum(1 for s in slots if s.occupied)
    free = total - occupied
    pct_occupied = round(100.0 * occupied / total, 1) if total else 0.0
    return {"total": total, "occupied": occupied, "free": free, "pct_occupied": pct_occupied}


def draw_slots(frame: np.ndarray, slots: List[ParkingSlot]) -> np.ndarray:
    out = frame.copy()
    for slot in slots:
        x1, y1, x2, y2 = slot.box
        color = (0, 0, 255) if slot.occupied else (0, 200, 0)  # BGR: red / green
        label = "OCCUPIED" if slot.occupied else "FREE"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"#{slot.id} {label}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def default_grid_slots(frame_w: int, frame_h: int, rows: int = 2, cols: int = 5,
                        margin: int = 20) -> List[Box]:
    """Generates an evenly-spaced grid of slot boxes over the frame — a
    quick-start layout so the demo works before a user draws real ROIs."""
    usable_w = frame_w - 2 * margin
    usable_h = frame_h - 2 * margin
    cell_w = usable_w // cols
    cell_h = usable_h // rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            x1 = margin + c * cell_w + 4
            y1 = margin + r * cell_h + 4
            x2 = x1 + cell_w - 8
            y2 = y1 + cell_h - 8
            boxes.append((x1, y1, x2, y2))
    return boxes
