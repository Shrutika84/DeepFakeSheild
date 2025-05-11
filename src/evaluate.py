import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.models import SimpleCNN
from src.preprocess import load_dataset


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.round(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:\n")
    from sklearn.metrics import classification_report

    print(classification_report(
        all_labels,
        all_preds,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    real_path = "../Dataset/real"
    fake_path = "../Dataset/fake"
    X_train, X_test, y_train, y_test = load_dataset(real_path, fake_path)

    X_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 1, 2)), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("../models/simple_cnn.pth", map_location=device))

    evaluate_model(model, test_loader, device)
