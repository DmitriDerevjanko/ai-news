import cv2
import subprocess
import numpy as np
import shlex
import re
import signal
import math
import time
from ultralytics import YOLO
import os

# === Local video file ===
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "time_square.mp4")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"❌ Видео не найдено: {VIDEO_PATH}")

# === Get video resolution ===
probe_cmd = [
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
    VIDEO_PATH
]
probe = subprocess.run(probe_cmd, capture_output=True, text=True)
match = re.search(r"(\d+)\s*x\s*(\d+)", probe.stdout)
width, height = (int(match.group(1)), int(match.group(2))) if match else (1280, 720)
print(f"🎥 Local video resolution: {width}x{height}")

# === Start ffmpeg in infinite loop (-stream_loop -1) + slow motion (3x) ===
cmd = (
    f'ffmpeg -stream_loop -1 -loglevel warning -re -fflags nobuffer -flags low_delay '
    f'-i {shlex.quote(VIDEO_PATH)} -vf "setpts=3*PTS" '  # 👈 замедление
    f'-f image2pipe -pix_fmt bgr24 -vcodec rawvideo -'
)
print("🚀 Launching ffmpeg (slow motion x3)...")
process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
frame_size = width * height * 3

# === Load YOLOv8n ===
print("🧠 Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")

last_positions = {}
PIXEL_TO_METER = 0.05
FPS = 15
print("▶️ Detection + tracking started (looping & slowed). Press Q to quit.")

try:
    while True:
        raw = process.stdout.read(frame_size)
        if len(raw) != frame_size:
            err = process.stderr.readline().decode(errors="ignore").strip()
            if err:
                print("⚠️", err)
            continue

        # === Read and resize ===
        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        frame_resized = cv2.resize(frame, (960, 540))
        original = frame_resized.copy()  # сохраняем оригинал

        # === YOLO detection ===
        results = model.track(
            frame_resized,
            persist=True,
            conf=0.1,
            imgsz=960,
            tracker="bytetrack.yaml",
            verbose=False,
            device="cpu"
        )

        annotated = results[0].plot() if results and len(results) else frame_resized
        person_count = 0

        if results and results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            xyxy = results[0].boxes.xyxy.cpu().numpy()

            for i, cls in enumerate(classes):
                x1, y1, x2, y2 = xyxy[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                obj_id = ids[i]

                if cls == 0:  # person
                    person_count += 1

                if cls in [2, 3, 5, 7]:  # vehicles
                    if obj_id in last_positions:
                        dx = cx - last_positions[obj_id][0]
                        dy = cy - last_positions[obj_id][1]
                        dist = math.hypot(dx, dy)
                        speed = dist * FPS * PIXEL_TO_METER
                        cv2.putText(
                            annotated,
                            f"{speed:.1f} m/s",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1
                        )
                    last_positions[obj_id] = (cx, cy)

        cv2.putText(
            annotated, f"People: {person_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # === Combine two videos side by side ===
        combined = np.hstack((original, annotated))

        # === Show both ===
        cv2.imshow("🗽 Times Square | Original (Left) + YOLOv8 Detection (Right)", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    process.send_signal(signal.SIGTERM)
    process.wait()
    cv2.destroyAllWindows()
    print("✅ Stream finished successfully.")
