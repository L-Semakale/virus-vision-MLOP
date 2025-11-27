import torch
import threading
import os

MODEL_PATH = "model/model.pt"

def retrain_model():
    dummy_model = {"status": "retrained"}
    os.makedirs("model", exist_ok=True)
    torch.save(dummy_model, MODEL_PATH)
    print("Model retrained and saved!")

def retrain_model_background():
    thread = threading.Thread(target=retrain_model)
    thread.start()
    return {"message": "Retraining started in background"}
