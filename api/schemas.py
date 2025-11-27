from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float


class RetrainResponse(BaseModel):
    status: str
