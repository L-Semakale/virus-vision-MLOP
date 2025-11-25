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
