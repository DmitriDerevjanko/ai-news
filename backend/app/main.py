from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2, subprocess, numpy as np, shlex, re, math, time, signal
from ultralytics import YOLO
from datetime import datetime

app = FastAPI(title="AI Live SmartCam")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Stream URL (Times Square) ===
CAMERA_URL = (
    "https://videos-3.earthcam.com/fecnetwork/hdtimes10.flv/playlist.m3u8?"
    "t=vBci5OreTDT5OVZWlrH3hFWPpk6y83Y18ohQ4H190JPIFxYos0EkV%2BMfx7l01dtY2Tp18y4PaRanNC4yHRhYNA%3D%3D"
    "&td=202510251426"
)

# === YOLOv8n ===
print("🧠 Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")

object_history = {}
PIXEL_TO_METER = 0.05
FPS = 15


@app.get("/api/health")
def health():
    return {"status": "ok"}


def generate_stream():
    """MJPEG stream identical to the standalone YOLOv8 script (people + vehicles + speed)."""
    print("🎥 Detecting stream resolution...")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            CAMERA_URL
        ],
        capture_output=True, text=True
    )
    match = re.search(r"(\d+)x(\d+)", probe.stdout)
    width, height = (int(match.group(1)), int(match.group(2))) if match else (1280, 720)
    frame_size = width * height * 3
    print(f"✅ Stream resolution: {width}x{height}")

    cmd = (
        f'ffmpeg -loglevel warning -re -fflags nobuffer -flags low_delay '
        f'-headers "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        f'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\r\n'
        f'Referer: https://www.earthcam.com/\r\nOrigin: https://www.earthcam.com\r\n" '
        f'-i {shlex.quote(CAMERA_URL)} -f image2pipe -pix_fmt bgr24 -vcodec rawvideo -'
    )

    print("🚀 Launching ffmpeg...")
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("▶️ Detection + tracking started (press Q in local mode to stop).")

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
                        if obj_id in object_history:
                            dx = cx - object_history[obj_id][0]
                            dy = cy - object_history[obj_id][1]
                            dist = math.hypot(dx, dy)
                            speed = dist * FPS * PIXEL_TO_METER
                            cv2.putText(
                                annotated,
                                f"{speed:.1f} m/s",
                                (int(x1), int(y1) - 10),
                                font, 0.5, (0, 255, 255), 1
                            )
                        object_history[obj_id] = (cx, cy)

            cv2.putText(
                annotated, f"People: {person_count}", (10, 30),
                font, 1, (0, 255, 0), 2
            )

            # Encode and yield as MJPEG frame
            _, jpeg = cv2.imencode(".jpg", annotated)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    except Exception as e:
        print("❌ Stream error:", e)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait()
        print("✅ Stream closed.")


@app.get("/api/video")
def video_feed():
    """Return MJPEG stream endpoint."""
    return Response(generate_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
