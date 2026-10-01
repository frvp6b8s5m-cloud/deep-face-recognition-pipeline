import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from facial_recognition.config import settings
from facial_recognition.embeddings.embedding_model import FaceEmbedder
from facial_recognition.detection.detector import FaceDetector
from facial_recognition.detection.face_alignment import align_face


def parse_args():
    parser = argparse.ArgumentParser(description="Run face recognition inference on a single image.")
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--gallery_dir", type=str, default=str(settings.processed_data_dir / "train"))
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    probe_path = Path(args.probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe image not found: {probe_path}")

    model = FaceEmbedder(embedding_dim=settings.embedding_dim, num_classes=1000)
    model.eval()

    detector = FaceDetector()
    image = __import__("cv2").imread(str(probe_path))
    if image is None:
        raise ValueError(f"Could not read image: {probe_path}")

    faces = detector.detect(image)
    if not faces:
        print("No faces detected in probe image.")
        return

    aligned = align_face(image, faces[0])
    embedding = model.embed(aligned)
    print(f"Detected face. Embedding dimension: {embedding.shape[0]}")
    print(f"Embedding preview: {embedding[:10]}")

    gallery = []
    for person_dir in sorted(Path(args.gallery_dir).iterdir()):
        if not person_dir.is_dir():
            continue
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = __import__("cv2").imread(str(image_path))
            if img is None:
                continue
            dets = detector.detect(img)
            if not dets:
                continue
            face = align_face(img, dets[0])
            gallery.append((person_dir.name, model.embed(face)))

    if not gallery:
        print("No gallery faces found.")
        return

    best_identity = None
    best_score = -1.0
    from facial_recognition.matching.similarity import cosine_similarity
    for identity, emb in gallery:
        score = cosine_similarity(embedding, emb)
        if score > best_score:
            best_score = score
            best_identity = identity

    print(f"Best match: {best_identity} with score {best_score:.4f}")


if __name__ == "__main__":
    main()
