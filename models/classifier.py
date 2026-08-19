"""Feature fusion and classification head (Section 3.4).

1024 -> 256 -> 64 -> 2, with BatchNorm + GELU + Dropout between layers,
matching the paper's fusion/classification block exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        hidden1: int = 256,
        hidden2: int = 64,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fusion(z)
        return self.classifier(x)
