# `fakeartdetector.model`

This module defines the CNN used for classifying images from the CIFAKE dataset.

## What this module does

- Provides `FakeArtClassifier`, a PyTorch Lightning `LightningModule`
- Implements a small CNN backbone + MLP classifier + 1-unit logit head
- Exposes a binary-classification forward pass that returns **raw logits**

## Model I/O

- Input: RGB images with shape $(B, 3, 32, 32)$
- Output: logits with shape $(B, 1)$

The model is trained/evaluated as a binary classifier by applying a sigmoid and thresholding:

```python
from torch import sigmoid

logits = model(images).squeeze(1)          # (B,)
preds = (sigmoid(logits) > 0.5).long()     # (B,)
```

## Loss and training notes

The module defines `self.criterium = torch.nn.BCEWithLogitsLoss()`.

If you use `BCEWithLogitsLoss`, targets must be floats (0.0/1.0) and match the logits shape. In this repository, the training script squeezes logits and casts targets:

```python
logits = model(images).squeeze(1)
loss = model.criterium(logits, targets.float())
```

## API reference

::: fakeartdetector.model.FakeArtClassifier
