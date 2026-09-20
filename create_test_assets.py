"""
create_test_assets.py
------------------------
Generates small SYNTHETIC placeholder images/videos into test_assets/ so
the app is runnable end-to-end without needing to download real footage.

IMPORTANT: these are simple drawn shapes (rectangles/lines), not real
photos or video frames. YOLOv8 was trained on real photos, so it will
generally NOT detect these synthetic shapes as "car" etc. — they exist to
exercise the app's plumbing (upload -> read frames -> draw overlays -> UI),
not to demonstrate real detection accuracy. Swap in real dashcam / parking
lot / CCTV footage to see the models actually work.

Run: python create_test_assets.py
"""
from pathlib import Path
import cv2
import numpy as np

OUT = Path("test_assets")
OUT.mkdir(exist_ok=True)

W, H = 640, 480


def make_parking_lot_image():
    img = np.full((H, W, 3), (60, 60, 60), dtype=np.uint8)  # asphalt gray
    # lane lines
    for x in range(20, W - 20, 128):
        cv2.line(img, (x, 20), (x, H - 20), (255, 255, 255), 2)
    # a few "parked cars" as colored rectangles
    car_slots = [(30, 40, 110, 100), (300, 40, 380, 100), (430, 260, 510, 320)]
    for (x1, y1, x2, y2) in car_slots:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 200), -1)
    cv2.putText(img, "SYNTHETIC parking lot (placeholder)", (15, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "sample_parking_lot.jpg"), img)


def make_traffic_video(filename: str, n_frames: int = 90, wrong_way_car: bool = True):
    path = str(OUT / filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25, (W, H))

    # two "vehicles": one going down the frame (correct), one going up (wrong-way)
    car1_y = 0.0
    car2_y = float(H)
    for i in range(n_frames):
        frame = np.full((H, W, 3), (40, 90, 40), dtype=np.uint8)  # road-ish green/gray
        cv2.rectangle(frame, (0, 0), (W, H), (70, 70, 70), 30)  # border = "road edge"

        car1_y += 4.5
        cv2.rectangle(frame, (150, int(car1_y)), (210, int(car1_y) + 40), (0, 165, 255), -1)
        cv2.putText(frame, "car A (correct dir)", (150, max(15, int(car1_y) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if wrong_way_car:
            car2_y -= 4.0
            cv2.rectangle(frame, (400, int(car2_y)), (460, int(car2_y) + 40), (200, 0, 0), -1)
            cv2.putText(frame, "car B (wrong dir)", (400, max(15, int(car2_y) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, f"SYNTHETIC traffic clip - frame {i+1}/{n_frames}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def make_road_litter_image():
    img = np.full((H, W, 3), (90, 90, 90), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (W, H), (60, 60, 60), 40)
    # small scattered shapes standing in for litter
    for (cx, cy, r, color) in [(120, 200, 12, (255, 255, 255)), (300, 350, 10, (0, 255, 255)),
                                 (480, 150, 14, (200, 200, 255))]:
        cv2.circle(img, (cx, cy), r, color, -1)
    cv2.putText(img, "SYNTHETIC road surface (placeholder)", (15, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "sample_road_litter.jpg"), img)


if __name__ == "__main__":
    make_parking_lot_image()
    make_traffic_video("sample_traffic_speed.mp4", n_frames=100, wrong_way_car=False)
    make_traffic_video("sample_traffic_wrongway.mp4", n_frames=100, wrong_way_car=True)
    make_road_litter_image()
    print(f"Synthetic test assets written to {OUT.resolve()}")
