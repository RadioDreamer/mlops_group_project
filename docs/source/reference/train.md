# Training

Training is driven by Hydra configs and executed with PyTorch Lightning's `pl.Trainer`, exposed via an Invoke task + Typer CLI.

## What training does

- Loads the processed CIFAKE tensors via `cifake()`
- Trains `FakeArtClassifier` using Lightning `pl.Trainer`
- Optimizer is configured via Hydra (default: Adam), loss: `BCEWithLogitsLoss`
- Supports precision control (e.g., `32`, `bf16-mixed`) and data loader `num_workers`
- Saves the trained model `state_dict` to `cfg.dataset.savedTo.path`
- Writes per-run artifacts in Hydra's output directory (config + metrics + checkpoints)

## Train the model

Run via the Invoke task (recommended):

```bash
uv run invoke train
```

Show available training options:

```bash
uv run invoke train --help
```

Examples:

```bash
# Switch optimizer from config
uv run invoke train --optimizer sgd

# Override precision and number of data loader workers
uv run invoke train --precision 32 --num-workers 0
uv run invoke train --precision bf16-mixed --epochs 5

# List available precision options (macOS/MPS compatible)
uv run invoke list_precisions

# List available config group choices
uv run invoke list-configs
```

Or run the module directly:

```bash
uv run python -m fakeartdetector.train --help
```

## Outputs

After training, outputs include:

- Model state dict at `cfg.dataset.savedTo.path` (e.g., `models/base_model.pth`)
- Hydra output directory: `outputs/<date>/<time>/`
  - `config_full.yaml`: full composed Hydra config for reproducibility
  - `artifacts.yaml`: paths to saved model, best checkpoint, logs, and settings (e.g., `num_workers`)
  - `lightning/version_0/metrics.csv`: CSV logs with `train_loss`, `train_acc`, `val_loss`, `val_acc`, etc.
  - `tensorboard/`: TensorBoard event files for visualization
  - `checkpoints/`: Lightning checkpoints (best model by `val_loss`)
  - `train_hydra.log`: Loguru training log
  - `profiler/`: Profiler output (if profiling enabled)

## Training details

- Device selection: CUDA → MPS → CPU
- Optimizer: configurable via `cfg.optimizer` (default: Adam)
- Loss: `BCEWithLogitsLoss`
- Precision: configurable via `experiment.hyperparameters.precision` (default: `32`). Common values: `32`, `bf16-mixed` (recommended on MPS/macOS). Note: `16-mixed` is not supported on Apple MPS.
- Data loading: set `experiment.hyperparameters.num_workers` or `--num-workers` to control parallelism. On macOS MPS, `0` workers is often fastest.
- Predictions for accuracy: `preds = (logits > 0).long()` (equivalent to sigmoid threshold at 0.5)

Note: Use `uv run invoke train --print-config` to print the resolved Hydra config used for the run.

### Profiling

Profile training performance using built-in profilers:

```bash
# Advanced profiler (recommended) - shows timing per Lightning hook
uv run invoke train --profiler advanced --epochs 1

# PyTorch profiler - generates Chrome trace (may hang on macOS)
uv run invoke train --profiler pytorch --epochs 1

# Simple profiler - lightweight wall-time tracking
uv run invoke train --profiler simple --epochs 1

# List available profiler options
uv run invoke list-configs
```

**Advanced Profiler Output:**

- Prints timing summary to terminal after training
- Saves to `outputs/<date>/<time>/profiler/advanced_profiler.txt`
- Shows time spent in `training_step`, `validation_step`, `optimizer_step`, etc.

**PyTorch Profiler Output:**

- Saves trace to `outputs/<date>/<time>/profiler/trace.json`
- View in Chrome: open `chrome://tracing` and load the trace file
- Note: May hang on macOS MPS; use advanced profiler instead

### Viewing Training Metrics with TensorBoard

Training automatically logs to TensorBoard. View real-time or historical metrics:

```bash
# View all training runs
uv run invoke tensorboard

# View specific run
uv run invoke tensorboard --logdir outputs/2026-01-10/14-31-25/tensorboard

# Custom port
uv run invoke tensorboard --port 6007
```

Then open `http://localhost:6006` in your browser.

**TensorBoard Views:**

- **SCALARS**: Training/validation loss and accuracy curves
- **HPARAMS**: Compare hyperparameters across runs
- **PROFILE**: GPU/CPU timeline (if using PyTorch profiler)

### Using saved statistics

You can print a summary from a finished run:

```bash
uv run invoke training-summary --output-dir outputs/2026-01-10/14-31-25
```

Or load metrics in Python:

```python
from fakeartdetector.training_stats import load_metrics

df = load_metrics("outputs/2026-01-10/14-31-25")
print(df[["train_loss","val_loss","train_acc","val_acc"]].dropna().tail())
```

## API reference

::: fakeartdetector.train.train_impl
