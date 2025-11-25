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
