import threading
import time
import torch

def retrain_model_background():
    thread = threading.Thread(target=retrain)
    thread.start()

def retrain():
    print("Retraining started...")

    # Simulate training time
    time.sleep(5)

    # Example: Save updated weights
    dummy_model = torch.nn.Linear(10, 2)
    torch.save(dummy_model, "model/model.pt")

    print("Retraining completed.")
