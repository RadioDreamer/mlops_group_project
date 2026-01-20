# FakeArtDetector

An MLOps project for classifying images as **Real** vs **AI-generated** using the CIFAKE dataset.

## What this project does

- Preprocesses CIFAKE into PyTorch tensors in `data/processed/`
- Trains a small CNN and saves a checkpoint to `models/model.pth`
- Evaluates the checkpoint on the CIFAKE test split
- Produces simple figures (training curves and embedding visualization)

## Quickstart

```bash
# Install dependencies
uv sync

# Download/process CIFAKE into data/processed/
uv run invoke preprocess-data

# Train and write a checkpoint to models/model.pth
uv run invoke train

# Evaluate the checkpoint on the test split
uv run invoke evaluate

# Visualize embeddings (writes a figure to reports/figures/)
uv run invoke visualize
```

## Key outputs

- `data/processed/train_images.pt`
- `data/processed/train_target.pt`
- `data/processed/test_images.pt`
- `data/processed/test_target.pt`
- `models/model.pth`
- `reports/figures/training_statistics.png`
- `reports/figures/embeddings.png` (default)

## Where to go next

- Setup and local run instructions: [Getting Started](getting-started.md)
- Day-to-day commands and artifacts: [Workflows](workflows.md)
- API reference: see the **API Reference** section in the navigation
