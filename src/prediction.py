from src.model import load_trained_model
from src.preprocessing import preprocess_single_image
import numpy as np




def predict_image(img_path, model_path="models/best_model.keras", label_encoder=None):
model = load_trained_model(model_path)
img = preprocess_single_image(img_path)


preds = model.predict(img)
class_index = np.argmax(preds)


if label_encoder:
return label_encoder.inverse_transform([class_index])[0], preds


return class_index, preds
