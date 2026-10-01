import cv2
import numpy as np


def align_face(image: np.ndarray, bbox: tuple, target_size: int = 160) -> np.ndarray:
    """Return a cropped, resized face image from a bounding box."""
    x, y, w, h = bbox
    face = image[y : y + h, x : x + w]
    if face.size == 0:
        raise ValueError("Invalid face bounding box.")
    face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return face
