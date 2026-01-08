# Workflows

## Data preprocessing

The data preprocessing script downloads the CIFAKE dataset from Hugging Face, converts images to tensors, and writes them into `data/processed/` as `.pt` files.

Run via Invoke:

```bash
uv run invoke preprocess-data
```

Outputs:

- `data/processed/train_images.pt`
- `data/processed/train_target.pt`
- `data/processed/test_images.pt`
- `data/processed/test_target.pt`

## Training

Training is currently scaffolded in `src/fakeartdetector/train.py`. The `invoke train` task runs that file:

```bash
uv run invoke train
```

If you extend training, this is the place to document:

- model architecture
- config/hyperparameters
- metrics + evaluation
- where artifacts are saved (e.g. `models/`)

## Docker images

Build the Docker images:

```bash
docker build -t train:latest . -f dockerfiles/train.dockerfile
docker build -t api:latest . -f dockerfiles/api.dockerfile
```

Or via Invoke:

```bash
uv run invoke docker-build
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
