# `fakeartdetector.visualize`

This module creates a 2D visualization of learned embeddings from a trained model on the CIFAKE test split.

## What this module does

- Loads a `FakeArtClassifier` checkpoint (`state_dict`)
- Loads the processed CIFAKE test tensors from `data/processed/`
- Computes per-image embeddings from the model (backbone + classifier output)
- Reduces embeddings to 2D using t-SNE and saves a scatter plot

## Create an embedding plot

Run via the Invoke task (recommended):

```bash
uv run invoke visualize
```

Or run the module directly:

```bash
uv run src/fakeartdetector/visualize.py models/model.pth
```

You can optionally choose the output filename:

```bash
uv run src/fakeartdetector/visualize.py models/model.pth --figure-name embeddings.png
```

The figure is saved to:

- `reports/figures/<figure_name>`

## Notes and assumptions

- Expected data files:
  - `data/processed/test_images.pt`
  - `data/processed/test_target.pt`
- Device selection: CUDA → MPS → CPU
- Embeddings are taken from `model.classifier(model.backbone(images))` (before the final 1-unit head)
- Dimensionality reduction:
  - PCA to 100 dims is only applied if the embedding dimension is greater than 500
  - t-SNE is then applied to produce 2D coordinates

## API reference

::: fakeartdetector.visualize.visualize
