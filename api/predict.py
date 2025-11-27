from fastapi import UploadFile
from api.model_loader import load_model, prepare_image
from api.schemas import PredictionResponse
import torch
import torch.nn.functional as F

# Your class labels (update these for your dataset)
CLASS_NAMES = ["Adenovirus", "Covid19", "Influenza", "Normal"]

async def predict_image(upload: UploadFile) -> PredictionResponse:
    model = load_model()
    image = prepare_image(upload.file)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)[0].tolist()

    predicted_index = int(torch.argmax(logits, dim=1))
    predicted_name = CLASS_NAMES[predicted_index]
    confidence = max(probs)

    return PredictionResponse(
        class_name=predicted_name,
        confidence=confidence
    )
