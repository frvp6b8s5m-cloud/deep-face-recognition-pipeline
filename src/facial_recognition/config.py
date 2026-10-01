from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    checkpoint_dir: Path = project_root / "checkpoints"
    log_dir: Path = project_root / "logs"
    device: str = "cpu"
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
    face_detector: str = "opencv"
    model_name: str = "face_embedder"


settings = Settings()
