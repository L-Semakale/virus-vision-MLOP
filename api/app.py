from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from src.preprocessing import preprocess_single_image
from src.prediction import predict_image
from src.retrain import trigger_retraining

app = FastAPI(title="Virus Classification API", description="ML Model Inference & Retraining API")

MODEL_PATH = "models/best_model.keras"
ENCODER_PATH = "models/label_encoder.pkl"
UPLOAD_DIR = "uploaded_data"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model & encoder at startup
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Predict the class of one image."""
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = preprocess_single_image(file_location)
    pred_class, conf = predict_image(model, encoder, img)

    return {"prediction": pred_class, "confidence": float(conf)}

@app.post("/upload-bulk")
def upload_bulk(files: list[UploadFile] = File(...)):
    """Upload multiple images to be included in retraining dataset."""
    saved_files = []
    for f in files:
        path = f"{UPLOAD_DIR}/{f.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append(path)
    return {"uploaded": saved_files, "count": len(saved_files)}

@app.post("/retrain")
def retrain():
    """Trigger model retraining using uploaded data."""
    new_model_path = trigger_retraining()
    return {"status": "Retraining complete", "model_saved_to": new_model_path}

@app.get("/health")
def health_check():
    return {"status": "API running", "model_loaded": bool(model)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
