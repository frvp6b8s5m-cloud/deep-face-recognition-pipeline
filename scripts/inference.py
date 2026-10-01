import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from facial_recognition.config import settings
from facial_recognition.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a facial recognition model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=str(settings.processed_data_dir))
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch_size", type=int, default=settings.batch_size)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(settings.seed)
    print("Training pipeline initialized.")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("This is a scaffold. Replace the training loop with your model training implementation.")


if __name__ == "__main__":
    main()
