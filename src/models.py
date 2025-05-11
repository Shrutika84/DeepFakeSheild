# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TransferXception(nn.Module):
    def __init__(self):
        super(TransferXception, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_model.classifier[1].in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)
