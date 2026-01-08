# `fakeartdetector.data`

This module contains the data preprocessing entrypoint for the CIFAKE dataset and small utilities for loading/visualizing the processed tensors.

## What this module does

- Downloads the CIFAKE dataset from Hugging Face
- Converts images to PyTorch tensors
- Saves processed tensors into `data/processed/` for later training/evaluation

## Preprocess the dataset

The preprocessing step writes four files:

- `data/processed/train_images.pt`
- `data/processed/train_target.pt`
- `data/processed/test_images.pt`
- `data/processed/test_target.pt`

Run via the Invoke task (recommended):

```bash
uv run invoke preprocess-data
```

Or run the module directly:

```bash
uv run src/fakeartdetector/data.py data/processed
```

Notes:

- Images are resized to $32\times32$ and converted to RGB.
- The preprocessing step currently does **not** apply normalization; it saves raw `ToTensor()` outputs in $[0, 1]$.

## Load processed tensors

Use `cifake()` to load the processed `.pt` files into `TensorDataset`s:

```python
from fakeartdetector.data import cifake

train_set, test_set = cifake()
```

## Visual sanity check

You can visualize a batch of images and their labels:

```python
import torch
from fakeartdetector.data import cifake, show_image_and_target

train_set, _ = cifake()
images, targets = next(iter(torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)))
show_image_and_target(images, targets)
```

## API reference

::: fakeartdetector.data.preprocess_data

::: fakeartdetector.data.cifake

::: fakeartdetector.data.normalize

::: fakeartdetector.data.show_image_and_target
