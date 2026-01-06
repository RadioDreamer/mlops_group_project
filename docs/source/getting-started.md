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

# Train (currently a placeholder)
uv run invoke train
```

## Build or serve the docs

```bash
# Build static site into ./build/
uv run invoke build-docs

# Serve locally (hot reload)
uv run invoke serve-docs
```
