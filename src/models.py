
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, n_mels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        h = self.net(x)  # [B,64,1,1]
        h = h.view(x.size(0), -1)
        return self.fc(h).squeeze(1)
