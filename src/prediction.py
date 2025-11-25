import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib

# ------------------------------------------------------------------
# Load Model + Encoder
# ------------------------------------------------------------------
def load_artifacts(model_path="./models/best_model.keras", encoder_path="./models/label_encoder.pkl"):
    model = load_model(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

# ------------------------------------------------------------------
# Predict Single Image
# ------------------------------------------------------------------
def predict_image(image_path, model, encoder, img_size=(128, 128)):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    class_name = encoder.inverse_transform([class_idx])[0]
    confidence = float(np.max(preds))

    return {
        "prediction": class_name,
        "confidence": round(confidence, 4)
    }
