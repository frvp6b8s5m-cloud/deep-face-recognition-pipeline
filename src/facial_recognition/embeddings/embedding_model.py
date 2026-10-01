from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn


class FaceEmbedder(nn.Module):
    """Small CNN-based embedding network for face recognition."""

    def __init__(self, embedding_dim: int = 512, num_classes: int = 1000):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.flatten(1)
        embedding = self.embedding(x)
        logits = self.classifier(embedding)
        return embedding, logits

    def embed(self, image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            embedding, _ = self(tensor)
        return embedding.squeeze(0).cpu().numpy()

    def embed_batch(self, images: Iterable[np.ndarray]) -> List[np.ndarray]:
        outputs = []
        for image in images:
            outputs.append(self.embed(image))
        return outputs
