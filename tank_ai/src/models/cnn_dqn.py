from __future__ import annotations

import torch
import torch.nn as nn


class CNNDQN(nn.Module):
    def __init__(self, input_channels: int = 6, output_size: int = 6, rows: int = 15, cols: int = 20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, rows, cols)
            flat = self.features(dummy).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
