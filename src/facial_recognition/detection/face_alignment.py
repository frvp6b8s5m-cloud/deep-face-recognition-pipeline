from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


class FaceDetector:
    """Basic face detector wrapper.

    This is a lightweight abstraction for using OpenCV Haar cascades or MTCNN
    if installed. The actual backend can be swapped by the user.
    """

    def __init__(self, backend: str = "opencv", scale_factor: float = 1.1, min_neighbors: int = 5):
        self.backend = backend
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self._cascade = None

        if backend == "opencv":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.backend == "opencv":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self._cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(30, 30),
            )
            return [tuple(map(int, box)) for box in detections]

        raise ValueError(f"Unsupported detector backend: {self.backend}")
