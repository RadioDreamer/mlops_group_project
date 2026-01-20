# Tasks & Invocations

This document explains how to use the project's Invoke tasks (the `tasks.py` file in the repository root). It lists every task, explains common options, and provides many examples and troubleshooting tips.

Table of contents

- Quickstart
- How Invoke is used in this repo
- Common patterns and environment notes
- Detailed task reference (examples and flags)
- Troubleshooting
- CI / Automation tips

---

## Quickstart

1. Install project dependencies using the repository package manager:

```bash
uv sync          # install runtime dependencies
uv sync --dev    # (optional) install developer dependencies
```

2. List available tasks:

```bash
inv --list
```

3. Run a task, for example training:

```bash
inv train --epochs 2 --lr 0.001
```

The `inv` CLI invokes the tasks defined in `tasks.py`.

---

## How Invoke is used in this repo

- `tasks.py` contains developer convenience tasks to run common workflows (training, evaluation, docs, docker builds, etc.).
- The tasks generally forward options to the project's Typer/Hydra CLIs (e.g., `python -m fakeartdetector.train`) or call external tools (e.g., `docker`, `dvc`, `streamlit`).
- Many tasks use the `uv run` prefix. `uv` is the project's packaging/runtime helper; if `uv` is not present in your environment, substitute the underlying command (e.g., run `python -m fakeartdetector.train` directly).

---

## Common patterns and environment notes

- On macOS, `num_workers` for data loaders is often set to `0` to avoid multiprocessing issues.
- Precision flags: the repo supports `32`, `bf16-mixed`, `bf16`, and `bf16-true` for training. Use `inv list_precisions` to see short descriptions.
- For local development of the API, use `inv startapi` and access `http://127.0.0.1:8000`.
- To run the Streamlit frontend locally, use `inv frontend` (optionally `--browser True`).

---

## Detailed task reference

All tasks live in `tasks.py`. Below are the commonly used tasks with their flags and examples.

- `preprocess-data`
  - Purpose: Run the data preprocessing pipeline.
  - Flags:
    - `--out-dir` (default: `data/processed`) — output directory for processed data.
  - Example:

```bash
inv preprocess-data --out-dir data/processed
```

- `startapi`
  - Purpose: Start the FastAPI server via `uvicorn` for local development.
  - Flags:
    - `--host` (default: `127.0.0.1`)
    - `--port` (default: `8000`)
  - Example:

```bash
inv startapi --host 0.0.0.0 --port 8000
```

- `train`
  - Purpose: Launch the training CLI (`fakeartdetector.train`) which is Hydra/Typer-backed.
  - Flags (selected): `--lr`, `--epochs`, `--batch-size`, `--num-workers`, `--precision`, `--profiler`, `--experiment`, `--dataset`, `--logging`, `--optimizer`, `--config-name`, `--print-config`.
  - Precision options: `32`, `bf16-mixed`, `bf16`, `bf16-true`. Use `inv list_precisions` to see descriptions.
  - Examples:

```bash
inv train --epochs 10 --lr 0.001 --batch-size 64
inv train --experiment trials1 --precision bf16-mixed
```

- `train-help`
  - Purpose: Print the CLI help for the training module (Typer/Hydra flags).
  - Example:

```bash
inv train-help
```

- `list-configs`
  - Purpose: List available config group options under `configs/` (experiment, dataset, logging, optimizer, profiler).
  - Example:

```bash
inv list-configs
```

- `list-precisions`
  - Purpose: Print supported precision choices for training.
  - Example:

```bash
inv list-precisions
```

- `tensorboard`
  - Purpose: Start TensorBoard pointing to the project's `outputs` directory or another logs directory.
  - Flags: `--logdir`, `--port`.
  - Example:

```bash
inv tensorboard --logdir outputs/2026-01-14/17-45-00 --port 6006
```

- `test`
  - Purpose: Run `pytest` with coverage.
  - Flags: `--pattern` (run a matching file or folder pattern; default `tests/`).
  - Examples:

```bash
inv test
inv test --pattern tests/test_api.py
```

- `frontend`
  - Purpose: Start the Streamlit frontend.
  - Flags: `--browser` (open browser after starting).
  - Example:

```bash
inv frontend --browser True
```

- `evaluate`
  - Purpose: Run evaluation CLI (`fakeartdetector.evaluate`).
  - Flags (selected): `--checkpoint`, `--batch-size`, `--threshold`, `--config-name`, `--dataset`, `--evaluate-cfg`, `--print-config`.
  - Example:

```bash
inv evaluate --checkpoint outputs/latest/model.ckpt --batch-size 64
```

- `visualize`
  - Purpose: Run embedding visualization CLI (`fakeartdetector.visualize`).
  - Flags: `--checkpoint`, `--figure-name`, `--output-dir`, `--data-dir`, `--batch-size`, `--pca-*`, `--tsne-*`, `--seed`.
  - Example:

```bash
inv visualize --checkpoint models/base_model.pth --figure-name embeddings.png
```

- `docker-build` / `docker-build-api`
  - Purpose: Build Docker images for training and the API.
  - Flags: `--progress` (docker build progress mode: `plain|auto|tty`).
  - Example:

```bash
inv docker-build --progress plain
inv docker-build-api --progress plain
```

- `dvc` / `pull-data`
  - `dvc`: Add a folder to DVC, commit, and push remote storage. Flags: `--folder`, `--message`.
  - `pull-data`: Run `dvc pull` to fetch artifacts.
  - Examples:

```bash
inv dvc --folder data/raw --message "add raw images"
inv pull-data
```

- `build-docs` / `serve-docs`
  - Purpose: Build or serve MkDocs documentation using `docs/mkdocs.yaml`.
  - Examples:

```bash
inv build-docs
inv serve-docs
```

- `make-req-txt`
  - Purpose: Export runtime and dev dependencies to `requirements.txt` and `requirements_dev.txt`.
  - Example:

```bash
inv make-req-txt
```

- `push`
  - Purpose: Add all changes, commit with a message, and push to the current branch (sets upstream if needed).
  - Flags: `--message`.
  - Example:

```bash
inv push --message "fix: update README"
```

---

## Troubleshooting

- If a task fails with `uv: command not found`, you can run the underlying command directly. For example, replace `uv run python -m fakeartdetector.train` with `python -m fakeartdetector.train`.
- On macOS, if multiprocessing data loaders crash, set `--num-workers 0` when training.
- If Docker builds fail on macOS with permissions, ensure you have the correct Docker context and resources.
- If Streamlit doesn't open the browser, start it with `inv frontend --browser True` or open the displayed URL manually.

---

## CI / Automation tips

- Use `inv test --pattern tests/test_api.py` to run only a focused API test in CI for quick feedback.
- Use `inv make-req-txt` in a job that regenerates pinned requirements after dependency changes.
- Use `inv build-docs` as a step to validate documentation builds before deployment.

---

If you'd like, I can also:

- Add this page to the MkDocs `nav` automatically.
- Generate a short reference cheat-sheet (single page) for common day-to-day commands.
