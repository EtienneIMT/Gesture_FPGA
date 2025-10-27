# models/simple_cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,128), nn.ReLU(),
            nn.Linear(128,n_classes)
        )
    def forward(self,x): return self.net(x)
