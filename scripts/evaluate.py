import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from facial_recognition.config import settings


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for a face recognition model.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Inference on image: {args.image_path}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {settings.device}")
    print("This scaffold expects a trained model and embedding extraction implementation.")


if __name__ == "__main__":
    main()
