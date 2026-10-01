import cv2
import numpy as np


def normalize_image(image: np.ndarray, target_size: tuple = (160, 160)) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image
