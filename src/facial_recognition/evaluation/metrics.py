from __future__ import annotations

from typing import Iterable, List

import numpy as np


def accuracy_score(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must match.")
    if not y_true:
        return 0.0
    return sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)


def precision_recall(y_true: Iterable[int], y_pred: Iterable[int]):
    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall
