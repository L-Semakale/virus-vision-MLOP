import torch
import threading
import os

MODEL_PATH = "model/model.pt"

def retrain_model():
    """Fake retraining function."""
    dummy_model = {"status": "retrained"}
    os.makedirs("model", exist_ok=True)
    torch.save(dummy_model, MODEL_PATH)
    print("Model retrained and saved!")

def retrain_model_background():
    """Runs retraining in a background thread."""
    thread = threading.Thread(target=retrain_model)
    thread.start()
    return {"message": "Retraining started in background"}
