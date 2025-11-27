from fastapi import UploadFile
from api.model_loader import load_model, prepare_image
from api.schemas import PredictionResponse
import torch.nn.functional as F

async def predict_image(upload: UploadFile) -> PredictionResponse:
    model = load_model()
    image = prepare_image(upload.file)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1).tolist()[0]

    predicted_class = probs.index(max(probs))

    return PredictionResponse(
        prediction=predicted_class,
        probabilities=probs
    )
