from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: List[float]

class RetrainResponse(BaseModel):
    status: str
