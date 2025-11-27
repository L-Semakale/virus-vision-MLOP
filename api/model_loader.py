import torch

MODEL_PATH = "model/model.pt"
model = None

def load_model():
    global model
    if model is None:
        model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        # No .eval() because this is NOT a neural network
    return model
