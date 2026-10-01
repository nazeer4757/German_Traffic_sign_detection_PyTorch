import io, os
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms
from app.classes import CLASS_NAMES

MODEL_PATH = os.getenv("MODEL_PATH", "models/weighted.pt")
THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))

app = FastAPI(title="Traffic Sign Classifier (GTSRB)")
_ck = torch.load(MODEL_PATH, map_location="cpu")
SIZE = _ck.get("size", 64)
_model = models.resnet18()
_model.fc = torch.nn.Linear(_model.fc.in_features, len(CLASS_NAMES))
_model.load_state_dict(_ck["state_dict"])
_model.eval()
_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.get("/health")
def health():
    return {"status": "ok", "classes": len(CLASS_NAMES), "input_size": SIZE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")
    with torch.no_grad():
        probs = torch.softmax(_model(_tf(img).unsqueeze(0)), 1)[0]
    top = probs.topk(3)
    conf = float(top.values[0])
    return {
        "prediction": CLASS_NAMES[int(top.indices[0])],
        "class_id": int(top.indices[0]),
        "confidence": conf,
        "uncertain": conf < THRESHOLD,
        "top3": [{"class": CLASS_NAMES[int(i)], "confidence": float(v)} for v, i in zip(top.values, top.indices)],
    }