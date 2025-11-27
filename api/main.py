from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from api.predict import predict_image
from api.retrain import retrain_model_background
from api.schemas import PredictionResponse, RetrainResponse

app = FastAPI(title="Virus Vision API", version="1.0")

# CORS (required for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    return await predict_image(file)

@app.post("/retrain", response_model=RetrainResponse)
async def retrain_endpoint():
    retrain_model_background()
    return {"status": "Retraining started in the background."}
