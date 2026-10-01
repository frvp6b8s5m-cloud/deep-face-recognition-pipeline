import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a folder-based face dataset for training.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--split", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        images = sorted(p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if not images:
            continue

        split_index = max(1, int(len(images) * (1.0 - args.split)))
        train_images = images[:split_index]
        test_images = images[split_index:]

        person_train_dir = train_dir / person_dir.name
        person_train_dir.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            target = person_train_dir / img.name
            if not target.exists():
                target.write_bytes(img.read_bytes())

        person_test_dir = test_dir / person_dir.name
        person_test_dir.mkdir(parents=True, exist_ok=True)
        for img in test_images:
            target = person_test_dir / img.name
            if not target.exists():
                target.write_bytes(img.read_bytes())

    print(f"Prepared dataset at: {output_dir}")
    print(f"Train samples: {sum(1 for _ in train_dir.rglob('*') if _.is_file())}")
    print(f"Test samples: {sum(1 for _ in test_dir.rglob('*') if _.is_file())}")


if __name__ == "__main__":
    main()
