# Training

Training is driven by Hydra configs and exposed via an Invoke task + Typer CLI.

## What training does

- Loads the processed CIFAKE tensors via `cifake()`
- Trains `FakeArtClassifier` using Adam and `BCEWithLogitsLoss`
- Saves the trained model `state_dict` to the configured `cfg.dataset.savedTo.path`
- Saves a training statistics figure to `reports/figures/training_statistics.png`

## Train the model

Run via the Invoke task (recommended):

```bash
uv run invoke train
```

Show available training options:

```bash
uv run invoke train --help
```

Example: switch optimizer from config:

```bash
uv run invoke train --optimizer sgd
```

List available config group choices:

```bash
uv run invoke list-configs
```

Or run the module directly:

```bash
uv run python -m fakeartdetector.train --help
```

## Outputs

After training, the script writes:

- Model checkpoint at `cfg.dataset.savedTo.path`
- `reports/figures/training_statistics.png` (loss and accuracy curves)

Additionally, each run writes a Loguru training log into Hydra's per-run output folder:

- `outputs/.../train_hydra.log`

## Training details

- Device selection: CUDA → MPS → CPU
- Optimizer: configurable via `cfg.optimizer` (default: Adam)
- Loss: `BCEWithLogitsLoss`
- Predictions for accuracy: `preds = (logits > 0).long()` (equivalent to sigmoid threshold at 0.5)

Note: Use `uv run invoke train --print-config` to print the resolved Hydra config used for the run.

## API reference

::: fakeartdetector.train.train_impl
