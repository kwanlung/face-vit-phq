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
import math
from itertools import compress
import textwrap

#--------------------config---------------------#
MODEL_PATH          = pathlib.Path("output/final_model.pth")
print(f"Loading model from {MODEL_PATH}")
LABELS: List[str]   = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
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
FACE_MODEL = str(pathlib.Path("models/face_detection_yunet_2023mar.onnx"))
face_net = cv2.FaceDetectorYN.create(
            FACE_MODEL, "", (320, 320), 0.6, 0.3,         # conf, NMS
          )
def detect_faces(img_bgr, conf_th=0.6):
    h, w = img_bgr.shape[:2]
    face_net.setInputSize((w, h))
    _, faces = face_net.detect(img_bgr)
    boxes = []
    if faces is not None:
        for x1, y1, w_, h_, conf, *_ in faces[faces[:,4] > conf_th]:
            boxes.append((int(x1), int(y1), int(w_), int(h_), float(conf)))
    return boxes

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
            for (x, y, w, h, det_conf) in detect_faces(frame):
                # Clamp to frame bounds just in case
                x, y = max(0,x), max(0,y)
                w = min(w, frame.shape[1]-x)
                h = min(h, frame.shape[0]-y)
                if w == 0 or h == 0:           # skip degenerate crop
                    continue
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                inp = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(inp)
                    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

                # build a compact probability string, top-3 only
                order = probs.argsort()[::-1][:3]
                prob_txt = " | ".join(f"{LABELS[i][:4]} {probs[i]*100:4.1f}%" for i in order)


                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # wrap long text onto two lines if necessary
                y0 = y - 10
                for line in textwrap.wrap(prob_txt, width=28):
                    cv2.putText(frame, line, (x, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    y0 -= 18                # line spacing                                 
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
        
        boxes = detect_faces(frame)
        if not boxes:
            print("No faces detected.")
            return
        
        for (x, y, w, h, det_conf) in detect_faces(frame):
                # Clamp to frame bounds just in case
                x, y = max(0,x), max(0,y)
                w = min(w, frame.shape[1]-x)
                h = min(h, frame.shape[0]-y)
                if w == 0 or h == 0:           # skip degenerate crop
                    continue
                face = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                inp = preprocess(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(inp)
                    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

                # build a compact probability string, top-3 only
                order = probs.argsort()[::-1][:3]
                prob_txt = " | ".join(f"{LABELS[i][:4]} {probs[i]*100:4.1f}%" for i in order)


                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # wrap long text onto two lines if necessary
                y0 = y - 10
                for line in textwrap.wrap(prob_txt, width=28):
                    cv2.putText(frame, line, (x, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    y0 -= 18 
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = CAPTURE_DIR / f"{ts}.jpg"
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