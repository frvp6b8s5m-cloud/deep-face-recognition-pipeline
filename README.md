# Deep Face Recognition Pipeline

A complete starter-to-production Python project for a deep learning facial recognition pipeline. This repository includes face detection, embedding extraction, similarity matching, dataset preparation, evaluation, and training/inference workflows.

## Features

- Face detection and bounding box extraction
- Face cropping and basic alignment
- CNN-based embedding extractor
- Similarity matching using cosine and Euclidean metrics
- Dataset preparation from folder structure
- Training loop and evaluation pipeline
- Command-line scripts for dataset prep, training, inference, and evaluation

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   ├── config.yaml
│   └── model_config.yaml
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── src/
│   └── facial_recognition/
│       ├── __init__.py
│       ├── config.py
│       ├── utils.py
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   └── face_alignment.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── embedding_model.py
│       ├── matching/
│       │   ├── __init__.py
│       │   ├── similarity.py
│       │   └── matcher.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── checkpoints/
├── logs/
├── tests/
│   └── __init__.py
└── .github/
```

## Installation

```bash
git clone https://github.com/frvp6b8s5m-cloud/deep-face-recognition-pipeline.git
cd deep-face-recognition-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset layout

This project expects a folder-based dataset with one directory per identity:

```text
data/raw/
├── person_001/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── person_002/
│   ├── img_001.jpg
│   └── ...
└── ...
```

Run dataset preparation:

```bash
python scripts/prepare_dataset.py --input_dir data/raw --output_dir data/processed --split 0.2
```

This creates a manifest and train/test partitions.

## Training

```bash
python scripts/train.py --data_dir data/processed --epochs 25 --batch_size 32
```

## Inference and recognition

```bash
python scripts/inference.py --probe_path path/to/probe.jpg --gallery_dir data/processed/train --model_path checkpoints/best_model.pth
```

## Evaluation

```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pth --data_dir data/processed/test
```

## Notes

- The system is implemented as a solid starter pipeline and can be upgraded to stronger architectures such as FaceNet, ArcFace, or ResNet-based embeddings.
- The default detector uses an OpenCV Haar cascade as a robust fallback for local development.
- For production-quality deployment, replace the lightweight CNN backbone with a pretrained face recognition model and larger dataset pipeline.

## License

MIT
