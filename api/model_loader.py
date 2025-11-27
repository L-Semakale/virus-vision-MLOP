import torch
from torchvision import transforms
from PIL import Image

MODEL_PATH = "model/model.pt"
model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_model():
    global model
    if model is None:
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
    return model

def prepare_image(file):
    img = Image.open(file).convert("RGB")
    return transform(img).unsqueeze(0)
