# `fakeartdetector.train`

This module trains the CNN on the processed CIFAKE dataset and writes a checkpoint and a simple training plot.

## What this module does

- Loads the processed CIFAKE tensors via `cifake()`
- Trains `FakeArtClassifier` using Adam and `BCEWithLogitsLoss`
- Saves the trained model `state_dict` to `models/model.pth`
- Saves a training statistics figure to `reports/figures/training_statistics.png`

## Train the model

Run via the Invoke task (recommended):

```bash
uv run invoke train
```

Or run the module directly:

```bash
uv run src/fakeartdetector/train.py
```

## Outputs

After training, the script writes:

- `models/model.pth` (PyTorch `state_dict` checkpoint)
- `reports/figures/training_statistics.png` (loss and accuracy curves)

## Training details

- Device selection: CUDA → MPS → CPU
- Optimizer: Adam
- Loss: `BCEWithLogitsLoss`
- Predictions for accuracy: `preds = (logits > 0).long()` (equivalent to sigmoid threshold at 0.5)

Note: the `train()` function signature accepts `lr`, `epochs`, and `batch_size`, but the current implementation uses a hard-coded dataloader batch size of 128.

## API reference

::: fakeartdetector.train.train
