import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare raw face dataset into a structured format.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--split", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset input: {input_dir}")
    print(f"Prepared output: {output_dir}")
    print(f"Validation split: {args.split}")
    print("This is the dataset preparation scaffold. Add your image labeling and partitioning logic here.")


if __name__ == "__main__":
    main()
