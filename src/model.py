# src/model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

def build_cnn(input_shape=(128,128,3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_trained_model(path="models/best_model.keras"):
    return load_model(path)


# src/preprocessing.py

import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_dataset(data_path, img_size=(128,128)):
    X, y = [], []
    classes = sorted(os.listdir(data_path))

    for label in classes:
        class_path = os.path.join(data_path, label)
        for file in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, file))
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)

    X = np.array(X) / 255.0
    le = LabelEncoder()
    y_encoded = to_categorical(le.fit_transform(y))
    return X, y_encoded, le


def preprocess_single_image(img_path, img_size=(128,128)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# src/prediction.py

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


# src/retrain.py

from src.model import build_cnn
from src.preprocessing import load_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
import os


def retrain(data_path="data/", save_path="models/retrained_model.keras", epochs=10):
    X, y, le = load_dataset(data_path)
    model = build_cnn(input_shape=X.shape[1:], num_classes=y.shape[1])

    os.makedirs("models", exist_ok=True)

    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[checkpoint]
    )

    return history, le

