import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def is_match(a: np.ndarray, b: np.ndarray, threshold: float = 0.6, metric: str = "cosine") -> bool:
    if metric == "cosine":
        score = cosine_similarity(a, b)
        return score >= threshold
    if metric == "euclidean":
        score = euclidean_distance(a, b)
        return score <= threshold
    raise ValueError(f"Unsupported metric: {metric}")
