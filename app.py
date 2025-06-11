import os, io, cv2, time, datetime, pathlib, threading
from typing import List

import cv2.data
import numpy as np
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

import torch
import torchvision.transforms as T

#--------------------config---------------------#
MODEL_PATH          = pathlib.Path("output/final_model.pth")
print(f"Loading model from {MODEL_PATH}")
LABELS: List[str]   = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
CAPTURE_DIR         = pathlib.Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

#--------------------laod model---------------------#
def load_model(path: pathlib.Path):
    from models.vit_model import EmotionViTClassifier 
    model = EmotionViTClassifier(model_name="vit_tiny_patch16_224",
                                 num_classes=len(LABELS),
                                 drop_rate=0.0)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model
model = load_model(MODEL_PATH)

preprocess = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

#--------------------OpenCV camera thread---------------------#

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Could not open video capture device")

# simple lock-protected global frame variable
g_latest_frame = None
g_lock = threading.Lock()

def camera_loop():
    global g_latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # flip horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        with g_lock:
            g_latest_frame = frame.copy()
# Start the camera thread
threading.Thread(target=camera_loop, daemon=True).start()

#--------------------face detection---------------------#
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_largest_face(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    # Find the largest face
    X, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (X, y, w, h)

#--------------------FastAPI app---------------------#
app = FastAPI(title="Live FER Demo")
templates = Jinja2Templates(directory="templates")

#--------------------HTML routes---------------------#
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video")
async def video_feed():
    def gen():
        while True:
            with g_lock:
                frame = g_latest_frame.copy() if g_latest_frame is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Detect face
            bbox = detect_largest_face(frame)
            if bbox:
                x, y, w, h = bbox
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                inp = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(inp)
                    label_idx = logits.argmax(1).item()
                label = LABELS[label_idx]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1/30)  # 30 FPS
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/capture")
async def capture_face(background_tasks: BackgroundTasks):
    "Capture current frame, annotate and save to ./captures"
    def _save():
        with g_lock:
            frame = g_latest_frame.copy() if g_latest_frame is not None else None
        if frame is None:
            return
        bbox = detect_largest_face(frame)
        label = "no_face"
        if bbox:
            x, y, w, h = bbox
            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            inp = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(inp)
                label_idx = logits.argmax(1).item()
            label = LABELS[label_idx]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CAPTURE_DIR / f"{ts}_{label}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Saved capture to {filename}")
    background_tasks.add_task(_save)
    return {"status": "Capture scheduled", "message": "Face capture will be saved in the background."}


# -----------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app",
                host="localhost",
                port=8000,
                reload=False)          # True if you want auto-reload
# -----------------------------------------------