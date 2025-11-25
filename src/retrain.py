import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from .preprocessing import load_dataset
from .model import build_model

MODEL_PATH = "./models/best_model.keras"
ENCODER_PATH = "./models/label_encoder.pkl"
TRAIN_DIR = "./data/train"

# ------------------------------------------------------------------
# Retrain model using newly uploaded data
# ------------------------------------------------------------------
def retrain_model(img_size=(128, 128), epochs=10, batch_size=16):
    print("Loading dataset...")
    X, y, encoder = load_dataset(TRAIN_DIR, img_size)

    num_classes = len(np.unique(y))
    input_shape = (img_size[0], img_size[1], 3)

    print("Building model...")
    model = build_model(input_shape, num_classes)

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    print("Training...")
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True
    )

    print("Saving label encoder...")
    joblib.dump(encoder, ENCODER_PATH)

    print("Retraining complete! Model updated.")
    return history
