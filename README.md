# 🧠 AI SmartCam — Live Object Detection

**AI SmartCam** is a real-time computer vision system built with **FastAPI**, **YOLOv8n**, and **OpenCV**.  
It processes a live HD stream from **EarthCam Times Square** and performs **real-time object detection and tracking** directly in the browser.

---

## 🚀 Overview

The system uses a lightweight **YOLOv8n** model for continuous detection of people, vehicles, and other visible objects.  
Frames are streamed from an external public camera and processed through:

ffmpeg → OpenCV → FastAPI (MJPEG)

The processed stream is displayed in a clean, modern HTML dashboard with smooth real-time visualization.

---

## ⚙️ Key Features

- 🧍 Real-time object detection using YOLOv8n  
- 🎯 Object tracking with ByteTrack  
- 🖥 Live annotated video stream served as MJPEG  
- ⚡ Optimized for low-latency CPU inference  
- 🌐 FastAPI backend + lightweight HTML frontend  
- 📷 Built on top of the public EarthCam Times Square feed  

---

## 🌐 Frontend Overview

The frontend is a single-page minimalist dashboard built with **HTML, CSS, and Vanilla JS**, designed for smooth live visualization.

Features:
- 🎥 Displays live annotated video from `/api/video`
- 💡 "About the System" section explaining model + pipeline
- 🧠 Clean responsive UI using Inter font and pastel blue gradient
- 🔗 Link to GitHub/source

Project layout (frontend):
- `frontend/`
  - `index.html` — main dashboard page

---

## 🧠 Backend Overview

The backend is powered by **FastAPI**, integrating **Ultralytics YOLOv8n** for detection and **ffmpeg** for live video capture.

Main components:
- FastAPI application with MJPEG streaming
- YOLOv8n model loaded via Ultralytics API
- `ffmpeg` subprocess piping frames to OpenCV
- Real-time object annotation and streaming response

Endpoints:
- `GET /api/health` → server health check  
- `GET /api/video` → live MJPEG video stream  

---

## 📂 Project Structure

High-level repository layout:

- `ai-smartcam/`
  - `backend/`
    - `main.py` — FastAPI app (YOLOv8 detection + MJPEG stream)
    - `requirements.txt` — backend dependencies
    - `utils/` (optional) — future helpers for analytics, speed, etc.
  - `frontend/`
    - `index.html` — dashboard UI

---

## 🧰 Installation & Run

1. Clone the repository  
   `git clone https://github.com/DmitriDerevjanko/ai-smartcam.git`

2. Go to backend  
   `cd ai-smartcam/backend`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Run FastAPI server  
   `uvicorn main:app --host 0.0.0.0 --port 8501`

5. Open the dashboard in your browser:  
   `http://localhost:8501`

---

## 📦 requirements.txt

Dependencies used by the backend:

- `fastapi==0.115.0`
- `uvicorn[standard]==0.30.6`
- `ultralytics>=8.3`
- `opencv-python>=4.10`
- `numpy>=1.26`

Python version: `3.12`

---

## 🖥 Example Output

Scene: 🗽 Times Square  
- People and vehicles detected in real time  
- Bounding boxes and labels drawn directly on the video stream  
- Served live through `/api/video` as MJPEG

---

## 🧩 Future Enhancements

Planned upcoming features:
- 🔢 Real-time object counting
- ⚡ Speed estimation for tracked vehicles
- 📊 Live analytics overlay
- 🧠 GPU acceleration support (CUDA / MPS)
- 💾 Optional recording and snapshot export

---

## 🔍 Tech Stack

| Layer             | Technology                     |
|-------------------|--------------------------------|
| Backend           | FastAPI, Python 3.12           |
| ML Model          | YOLOv8n (Ultralytics)          |
| Tracking          | ByteTrack                      |
| Video Processing  | OpenCV + ffmpeg                |
| Frontend          | HTML + CSS + JS                |
| Deployment        | Uvicorn / Docker-ready         |

---

## 👨‍💻 Author

**Dmitri Derevjanko**  
🎓 AI Systems & Computer Vision  
🌐 dmitriderevjanko.com  
🐙 GitHub

---

## 🪪 License

MIT License  
© 2025 — AI Smart Systems
