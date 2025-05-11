# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models import SimpleCNN
from src.preprocess import load_dataset


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(1, 11):  # 10 epochs
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    real_path = "../Dataset/real"
    fake_path = "../Dataset/fake"
    X_train, X_test, y_train, y_test = load_dataset(real_path, fake_path)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(np.transpose(X_train, (0, 3, 1, 2)), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")
    train_model(model, train_loader, criterion, optimizer, device)

    # Save the model
    torch.save(model.state_dict(), "../models/simple_cnn.pth")
    print("Model saved at ../models/simple_cnn.pth")