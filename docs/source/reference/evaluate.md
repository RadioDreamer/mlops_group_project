# `fakeartdetector.evaluate`

This module evaluates a saved model checkpoint on the CIFAKE test split.

## What this module does

- Loads a `FakeArtClassifier` and applies a saved `state_dict`
- Loads the processed CIFAKE tensors via `cifake()`
- Runs inference on the test set and prints accuracy
- Selects the best available device in this order: CUDA → MPS → CPU

## Evaluate a checkpoint

The evaluation entrypoint expects a path to a PyTorch `state_dict` checkpoint (as saved by `torch.save(model.state_dict(), ...)`).

Run via the Invoke task (recommended):

```bash
uv run invoke evaluate
```

Or run the module directly:

```bash
uv run src/fakeartdetector/evaluate.py models/model.pth
```

## How predictions are computed

The model outputs logits with shape $(B, 1)$. Evaluation converts logits to class predictions using a sigmoid threshold at 0.5:

```python
y_pred = (sigmoid(model(img).squeeze(1)) > 0.5).long()
```

Accuracy is computed as the fraction of correct predictions over the full test set.

## API reference

::: fakeartdetector.evaluate.evaluate
