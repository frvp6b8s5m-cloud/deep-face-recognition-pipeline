from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Folder-based dataset where each subdirectory is an identity."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[str, Path]]:
        samples = []
        for label_dir in sorted(self.root_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for image in sorted(label_dir.iterdir()):
                if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    samples.append((label_dir.name, image))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label, path = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
