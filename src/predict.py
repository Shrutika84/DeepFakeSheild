# src/predict.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from src.models import SimpleCNN

IMG_SIZE = 128


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return torch.tensor(img, dtype=torch.float32)


def predict(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(output).item()
        confidence = output.item()

    label = "Fake" if prediction == 1 else "Real"
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    return label, confidence


if __name__ == "__main__":
    test_image_path = "../Dataset/fake/fake1.jpg"  # Change this to your test image
    model_path = "../models/simple_cnn.pth"
    predict(test_image_path, model_path)
