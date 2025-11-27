import torch
import torch.nn.functional as F
from fastapi import UploadFile
from api.model_loader import load_model, prepare_image
from api.schemas import PredictionResponse

# Modify these based on your dataset
CLASS_NAMES = ["NORMAL", "COVID", "PNEUMONIA"]

async def predict_image(file: UploadFile) -> PredictionResponse:
    # Load model
    model = load_model()

    # Preprocess input image
    image_tensor = prepare_image(file.file)

    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()

    # Find best class
    best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    class_name = CLASS_NAMES[best_idx]
    confidence = float(probs[best_idx])

    # Return response in the NEW format
    return PredictionResponse(
        class_name=class_name,
        confidence=confidence
    )
