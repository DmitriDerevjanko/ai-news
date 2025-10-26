import cv2
import subprocess
import numpy as np
import shlex
import re
import signal
import math
import time
from ultralytics import YOLO

URL = "https://videos-3.earthcam.com/fecnetwork/hdtimes10.flv/playlist.m3u8?t=vBci5OreTDT5OVZWlrH3hFWPpk6y83Y18ohQ4H190JPIFxYos0EkV%2BMfx7l01dtY2Tp18y4PaRanNC4yHRhYNA%3D%3D&td=202510251426"

headers = (
    "-headers",
    "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36\r\n"
    "Referer: https://www.earthcam.com/\r\n"
    "Origin: https://www.earthcam.com\r\n"
)

# Get stream resolution
probe_cmd = [
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0"
] + list(headers) + [URL]
probe = subprocess.run(probe_cmd, capture_output=True, text=True)
match = re.search(r"(\d+)\s*x\s*(\d+)", probe.stdout)
width, height = (int(match.group(1)), int(match.group(2))) if match else (1280, 720)
print(f"🎥 Stream resolution: {width}x{height}")

# Start ffmpeg
cmd = (
    f'ffmpeg -loglevel warning -re -fflags nobuffer -flags low_delay '
    f'-headers "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
    f'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\r\n'
    f'Referer: https://www.earthcam.com/\r\nOrigin: https://www.earthcam.com\r\n" '
    f'-i {shlex.quote(URL)} -f image2pipe -pix_fmt bgr24 -vcodec rawvideo -'
)
print("🚀 Launching ffmpeg...")
process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
frame_size = width * height * 3

# Load YOLOv8n
print("🧠 Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")

last_positions = {}
PIXEL_TO_METER = 0.05
FPS = 15
print("▶️ Detection + tracking started. Press Q to quit.")

try:
    while True:
        raw = process.stdout.read(frame_size)
        if len(raw) != frame_size:
            err = process.stderr.readline().decode(errors="ignore").strip()
            if err:
                print("⚠️", err)
            continue

        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        frame_resized = cv2.resize(frame, (960, 540))

        # 👉 same YOLOv8n, just lower conf and higher imgsz
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

        cv2.imshow("🗽 Times Square — YOLOv8 Tracking", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    process.send_signal(signal.SIGTERM)
    process.wait()
    cv2.destroyAllWindows()
    print("✅ Stream finished successfully.")
