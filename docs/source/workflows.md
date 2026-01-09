# Workflows

This page collects the common end-to-end commands for running the project locally.

## Data preprocessing

Preprocessing downloads CIFAKE from Hugging Face, converts images to tensors, and saves them to `data/processed/`.

Run via Invoke (recommended):

```bash
uv run invoke preprocess-data
```

Outputs:

- `data/processed/train_images.pt`
- `data/processed/train_target.pt`
- `data/processed/test_images.pt`
- `data/processed/test_target.pt`

## Training

Training loads the processed tensors, trains `FakeArtClassifier`, saves a checkpoint, and writes a simple training figure.

Run via Invoke:

```bash
uv run invoke train
```

Outputs:

- `models/model.pth`
- `reports/figures/training_statistics.png`

## Evaluation

Evaluation loads `models/model.pth` (a PyTorch `state_dict`), runs inference on the CIFAKE test split, and prints test accuracy.

Run via Invoke:

```bash
uv run invoke evaluate
```

## Visualization

Visualization computes embeddings from the trained model on the CIFAKE test split, reduces them to 2D (t-SNE), and saves a scatter plot.

Run via Invoke:

```bash
uv run invoke visualize
```

Outputs (default):

- `reports/figures/embeddings.png`

## Tests

Run the test suite with coverage:

```bash
uv run invoke test
```

## Docker images

Build the Docker images via Invoke:

```bash
uv run invoke docker-build
```

Or run Docker directly:

```bash
docker build -t train:latest . -f dockerfiles/train.dockerfile
docker build -t api:latest . -f dockerfiles/api.dockerfile
```

## Docs

Serve docs locally:

```bash
uv run invoke serve-docs
```

Build docs into `./build/`:

```bash
uv run invoke build-docs
```
