# `ONNX Export`

This project includes tools to export the trained PyTorch Lightning model to ONNX format for inference with ONNX Runtime.

## Purpose

- Create a portable representation of the model for low-latency or cross-platform inference.

## How to export

- See `src/fakeartdetector/export_onnx.py` for the export script.
- Typical usage (example):

```bash
python src/fakeartdetector/export_onnx.py --checkpoint ./staged_model_dir/model.ckpt --output model.onnx
```

Note: there is no dedicated Invoke task for ONNX export at the moment — run the script directly as shown above, or I can add an `invoke` task if you prefer a consistent `inv` workflow.

## Notes & compatibility

- Ensure the model is in evaluation mode prior to export.
- ONNX ops supported depend on the PyTorch and ONNX versions; test the exported model with ONNX Runtime.

## Running ONNX model

Install dependencies via the project's package manager and run ONNX Runtime checks:

```bash
uv sync
uv run python -c "import onnxruntime as ort; print(ort.get_device())"
```

Then write a small runner that loads `model.onnx` and runs inference on preprocessed tensors (run it via `uv run`).
