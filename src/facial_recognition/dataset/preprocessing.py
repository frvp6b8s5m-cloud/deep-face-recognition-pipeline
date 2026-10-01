from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Path]:
        return sorted([p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        path = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return str(path), image
