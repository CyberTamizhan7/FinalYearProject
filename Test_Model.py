import os
import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Load the model ===
model = load_model("Chess_22_20.h5")

# === Load class labels ===
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# === Load and preprocess test image ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))  # Resize to match training size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = img_to_array(img) / 255.0              # Normalize
    img = np.expand_dims(img, axis=0)            # Add batch dimension
    return img

# === Predict function ===
def predict_piece(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_label = class_labels[class_index]
    return predicted_label, confidence * 100

# === Test it ===
# Replace with your own test image path
test_image_path = r"C:\Users\admin\Desktop\Final Year Project\Test Images\board722_a5.jpg"
print("Original Image : Black Knight")
predicted_piece, accuracy = predict_piece(test_image_path)
print(f"Predicted Piece: {predicted_piece}")
print(f"Confidence: {accuracy:.2f}%")

test_image_path = r"C:\Users\admin\Desktop\Final Year Project\Test Images\board247_c1.jpg"
print("Original Image : Black Knight")
predicted_piece, accuracy = predict_piece(test_image_path)
print(f"Predicted Piece: {predicted_piece}")
print(f"Confidence: {accuracy:.2f}%")


