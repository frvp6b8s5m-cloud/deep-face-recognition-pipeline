import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a facial recognition model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Evaluating model: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print("Evaluation scaffold is ready. Connect this to your metrics pipeline for accuracy, ROC, and verification metrics.")


if __name__ == "__main__":
    main()
