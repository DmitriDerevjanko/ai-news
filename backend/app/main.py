from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, subprocess, numpy as np, shlex, re, math, signal, os, time
from ultralytics import YOLO

app = FastAPI(title="🧠 AI SmartCam — Dual Stream")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Local fallback video ===
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "time_square.mp4")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"❌ Video not found: {VIDEO_PATH}")

print("🧠 Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")

object_history = {}
PIXEL_TO_METER = 0.05
FPS = 15


@app.get("/api/health")
def health():
    """Healthcheck endpoint"""
    return {"status": "ok"}


def generate_stream():
    """Dual MJPEG stream — original (left) + YOLO-annotated (right)."""
    print("🎥 Probing local video resolution...")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            VIDEO_PATH
        ],
        capture_output=True, text=True
    )

    match = re.search(r"(\d+)\s*x\s*(\d+)", probe.stdout)
    width, height = (int(match.group(1)), int(match.group(2))) if match else (1280, 720)
    frame_size = width * height * 3
    print(f"✅ Local video resolution: {width}x{height}")

    # === FFmpeg with slow motion (2× slower) and infinite loop ===
    cmd = (
        f'ffmpeg -stream_loop -1 -loglevel warning -re -fflags nobuffer -flags low_delay '
        f'-i {shlex.quote(VIDEO_PATH)} -vf "setpts=2*PTS" '
        f'-f image2pipe -pix_fmt bgr24 -vcodec rawvideo -'
    )

    print("🚀 Launching ffmpeg (slow motion ×2)...")
    process = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("▶️ Detection + tracking started (looping).")

    try:
        while True:
            raw = process.stdout.read(frame_size)
            if len(raw) != frame_size:
                # Если кадры закончились — возможно, ffmpeg перезапускается
                time.sleep(0.1)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
            frame_resized = cv2.resize(frame, (640, 360))
            original = frame_resized.copy()

            # --- YOLO detection ---
            results = model.track(
                frame_resized,
                persist=True,
                conf=0.15,
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

                    # Vehicle tracking + speed
                    if cls in [2, 3, 5, 7]:
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

            # Add people count text
            cv2.putText(
                annotated, f"People: {person_count}", (10, 30),
                font, 1, (0, 255, 0), 2
            )

            # Combine original + annotated side by side
            combined = np.hstack((original, annotated))

            # Encode as JPEG and yield frame
            _, jpeg = cv2.imencode(".jpg", combined)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    except Exception as e:
        print(f"❌ Stream error: {e}")
    finally:
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=2)
        except Exception:
            pass
        print("✅ Stream closed.")


@app.get("/api/video")
def video_feed():
    """Return MJPEG stream with side-by-side comparison."""
    return StreamingResponse(
        generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
