from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    checkpoint_dir: Path = data_dir / "models"
    log_dir: Path = project_root / "logs"
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    face_detection_threshold: float = 0.9
    similarity_threshold: float = 0.6
    image_size: int = 160
    embedding_dim: int = 512
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    seed: int = 42
    dataset_split: float = 0.2
    margin: float = 20
    embedding_metric: str = "cosine"
    face_detector: str = "mtcnn"
    model_name: str = "resnet_face_embedder"


settings = Settings()


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    num_workers: int = 4
    save_top_k: int = 3


training_config = TrainingConfig()
