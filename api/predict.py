import torch
import torch.nn.functional as F
from fastapi import UploadFile
from api.model_loader import load_model, prepare_image
from api.schemas import PredictionResponse

# Load model ONCE (not on every request)
MODEL = load_model()
CLASS_NAMES = ["Coronavirus", "Healthy Lung", "Tuberculosis"]  # <-- UPDATE

async def predict_image(upload: UploadFile) -> PredictionResponse:
    # Read image bytes
    contents = await upload.read()

    # Preprocess
    image = prepare_image(contents)

    # Inference
    with torch.no_grad():
        logits = MODEL(image)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Get class + confidence
    idx = int(probs.argmax())
    predicted_class = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return PredictionResponse(
        class_name=predicted_class,
        confidence=confidence
    )
