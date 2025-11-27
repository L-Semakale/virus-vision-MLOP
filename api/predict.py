from fastapi import UploadFile
from api.model_loader import load_model
from api.schemas import PredictionResponse

async def predict_image(file: UploadFile):

    # Load dummy model (dictionary)
    model = load_model()

    # Fake prediction result
    class_name = "Normal"
    confidence = 0.95

    return PredictionResponse(
        class_name=class_name,
        confidence=confidence
    )
