"""
app.py
FastAPI backend for Virus Image Classification
----------------------------------------------

Features:
- Predict a single uploaded image
- Upload multiple images for retraining
- Trigger model retraining
- Health + uptime endpoints
"""

import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from PIL import Image
import io
import joblib
import tensorflow as tf

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "./models/best_model.keras"
ENCODER_PATH = "./models/label_encoder.pkl"
RETRAIN_DATA_PATH = "./data/retrain/"

# Ensure retrain folder exists
os.makedirs(RETRAIN_DATA_PATH, exist_ok=True)

# -----------------------------
# Load model and encoder
# -----------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Model & encoder loaded successfully.")
except:
    model = None
    label_encoder = None
    print("⚠️ Warning: Model not loaded. Retraining required.")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Virus Image Classification API",
    description="API for prediction, retraining, and monitoring.",
    version="1.0.0"
)

start_time = datetime.now()


# -----------------------------
# Utility: Preprocess Image
# -----------------------------
def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((128, 128))
        arr = np.array(image) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")


# -----------------------------
# Endpoint: Healthcheck
# -----------------------------
@app.get("/health")
def health():
    uptime = datetime.now() - start_time
    return {
        "status": "running",
        "uptime": str(uptime),
        "model_loaded": model is not None
    }


# -----------------------------
# Endpoint: Predict
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    image_bytes = await file.read()
    img = preprocess_image(image_bytes)

    preds = model.predict(img)[0]
    class_idx = np.argmax(preds)
    class_name = label_encoder.inverse_transform([class_idx])[0]
    confidence = float(np.max(preds))

    return {
        "filename": file.filename,
        "predicted_class": class_name,
        "confidence": confidence
    }


# -----------------------------
# Endpoint: Upload Images for Retraining
# -----------------------------
@app.post("/upload-retrain-data")
async def upload_retrain_data(files: list[UploadFile] = File(...)):
    saved_files = []

    for file in files:
        contents = await file.read()
        save_path = os.path.join(RETRAIN_DATA_PATH, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)
        saved_files.append(file.filename)

    return {
        "message": "Files uploaded successfully.",
        "files_saved": saved_files
    }


# -----------------------------
# Endpoint: Trigger Retraining
# -----------------------------
@app.post("/retrain")
def retrain_model():
    """
    Calls your retraining pipeline.
    You will implement retraining inside src/retrain.py
    """
    try:
        from src.retrain import retrain_pipeline

        result = retrain_pipeline()
        return {"message": "Retraining completed.", "details": result}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Retraining failed: {str(e)}"}
        )


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
