# Getting started

## Prerequisites

- Python $\ge$ 3.11
- `uv` (recommended) or another environment manager

## Install

From the repository root:

```bash
uv sync
```

## Run the common tasks

This project uses `invoke` tasks defined in `tasks.py`.

```bash
# Run unit tests
uv run invoke test

# Preprocess/download dataset into data/processed/
uv run invoke preprocess-data

# Train
uv run invoke train

# See training options
uv run invoke train --help

# List available Hydra config groups
uv run invoke list-configs

# Evaluate (uses cfg.evaluate.model_checkpoint by default)
uv run invoke evaluate

# See evaluation options
uv run invoke evaluate --help
```

## Build or serve the docs

```bash
# Build static site into ./build/
uv run invoke build-docs

# Serve locally (hot reload)
uv run invoke serve-docs
```
