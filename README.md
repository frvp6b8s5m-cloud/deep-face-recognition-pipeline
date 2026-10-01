# Deep Face Recognition Pipeline

A deep learning-based facial recognition system designed for face detection, embedding extraction, similarity matching, and evaluation.

## Overview

This project provides a starter pipeline for building a facial recognition system with:

- face detection using MTCNN or Haar cascade
- face alignment and preprocessing
- embedding extraction using a CNN backbone
- similarity matching with cosine or Euclidean distance
- dataset utilities and evaluation scripts

## Repository Structure

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
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── prepare_dataset.py
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
│       │   └── similarity.py
│       ├── dataset/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       └── evaluation/
│           ├── __init__.py
│           └── metrics.py
└── tests/
    └── __init__.py
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

## Configuration

Edit the files in `configs/` to adjust:

- dataset paths
- model path
- detection threshold
- training hyperparameters
- similarity cutoff

## Usage

### Prepare dataset

```bash
python scripts/prepare_dataset.py --input_dir data/raw --output_dir data/processed --split 0.2
```

### Train model

```bash
python scripts/train.py --config configs/config.yaml
```

### Run inference

```bash
python scripts/inference.py --image_path path/to/image.jpg --model_path checkpoints/best_model.pth --config configs/config.yaml
```

### Evaluate model

```bash
python scripts/evaluate.py --model_path checkpoints/best_model.pth --data_dir data/processed/test --config configs/config.yaml
```

## Notes

This is a production-style starter project scaffold. You can swap in different backbones such as:

- Facenet
- ArcFace
- ResNet-based face embeddings
- MobileFaceNet

The code is structured to allow extension for real-world training and deployment.

## License

MIT
