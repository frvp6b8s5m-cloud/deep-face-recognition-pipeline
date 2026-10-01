import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from facial_recognition.config import settings
from facial_recognition.detection.detector import FaceDetector
from facial_recognition.detection.face_alignment import align_face
from facial_recognition.embeddings.embedding_model import FaceEmbedder
from facial_recognition.matching.similarity import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained face recognition model.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default=str(settings.processed_data_dir / "test"))
    return parser.parse_args()


def evaluate_model(model, data_dir: Path):
    detector = FaceDetector()
    results = []

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            faces = detector.detect(image)
            if not faces:
                continue
            aligned = align_face(image, faces[0], target_size=160)
            embedding = model.embed(aligned)
            results.append((person_dir.name, embedding))

    if not results:
        print("No test faces found.")
        return 0.0

    total = 0
    correct = 0
    for identity_a, emb_a in results:
        for identity_b, emb_b in results:
            if identity_a == identity_b:
                total += 1
                correct += 1 if cosine_similarity(emb_a, emb_b) >= 0.5 else 0
    return correct / total if total else 0.0


def main():
    args = parse_args()
    model = FaceEmbedder(embedding_dim=settings.embedding_dim, num_classes=1000)
    state = __import__("torch").load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    accuracy = evaluate_model(model, Path(args.data_dir))
    print(f"Evaluation accuracy proxy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
