from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn


class FaceEmbedder(nn.Module):
    """Simple CNN-style embedding backbone for a face recognition pipeline."""

    def __init__(self, embedding_dim: int = 512, num_classes: int = 1000):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, embedding_dim)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        return self.head(x)

    def embed(self, images: Iterable[np.ndarray]) -> List[np.ndarray]:
        outputs = []
        for image in images:
            tensor = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                embedding = self.forward(tensor)
            outputs.append(embedding.squeeze(0).cpu().numpy())
        return outputs
