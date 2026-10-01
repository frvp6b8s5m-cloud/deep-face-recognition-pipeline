from __future__ import annotations

from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class FaceMatcher:
    def __init__(self, threshold: float = 0.6, metric: str = "cosine"):
        self.threshold = threshold
        self.metric = metric

    def match(self, embedding_a: np.ndarray, embedding_b: np.ndarray):
        if self.metric == "cosine":
            score = cosine_similarity(embedding_a, embedding_b)
            return score, score >= self.threshold
        if self.metric == "euclidean":
            score = np.linalg.norm(np.asarray(embedding_a) - np.asarray(embedding_b))
            return score, score <= self.threshold
        raise ValueError(f"Unsupported metric: {self.metric}")
