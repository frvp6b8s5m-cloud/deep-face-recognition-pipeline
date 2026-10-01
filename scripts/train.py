import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from facial_recognition.config import settings
from facial_recognition.dataset.loader import ImageDataset
from facial_recognition.embeddings.embedding_model import FaceEmbedder


def parse_args():
    parser = argparse.ArgumentParser(description="Train a face recognition model on a labeled folder dataset.")
    parser.add_argument("--data_dir", type=str, default=str(settings.processed_data_dir))
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch_size", type=int, default=settings.batch_size)
    parser.add_argument("--learning_rate", type=float, default=settings.learning_rate)
    return parser.parse_args()


def make_label_map(dataset_dir: Path):
    labels = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    return {label: idx for idx, label in enumerate(labels)}


def collate_fn(batch):
    imgs, labels = zip(*batch)
    transformed = []
    out_labels = []
    for img, label in zip(imgs, labels):
        rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (160, 160))
        rgb = rgb.astype(np.float32) / 255.0
        rgb = torch.tensor(rgb).permute(2, 0, 1)
        transformed.append(rgb)
        out_labels.append(label)
    return torch.stack(transformed), torch.tensor([int(label) for label in out_labels])


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")

    labels = make_label_map(train_dir)
    dataset = ImageDataset(str(train_dir), transform=None)
    model = FaceEmbedder(embedding_dim=settings.embedding_dim, num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(settings.device)
    model.to(device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels_batch in loader:
            images = [cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) for img in images]
            tensor_batch = []
            label_batch = []
            for img, label in zip(images, labels_batch):
                img = cv2.resize(img, (160, 160)).astype(np.float32) / 255.0
                tensor_batch.append(torch.tensor(img).permute(2, 0, 1))
                label_batch.append(labels[label])
            x = torch.stack(tensor_batch).to(device)
            y = torch.tensor(label_batch, dtype=torch.long).to(device)
            optimizer.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {total_loss / max(len(loader), 1):.4f}")

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoints_dir / "best_model.pth")
    print(f"Model saved to {checkpoints_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
