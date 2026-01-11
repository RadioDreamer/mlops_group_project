from __future__ import annotations

from typing import Any

from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim, randn


class FakeArtClassifier(LightningModule):
    """CNN for classifying AI-generated vs human-created art (CIFAKE dataset)

    Assumptions
    -----------
        - Input images are RGB with shape (3, 32, 32).
        - The task is binary classification with two mutually exclusive classes.
        - The model outputs raw logits suitable for `CrossEntropyLoss`.

    Architecture:
        - Input: RGB image (3 × 32 × 32)

        - Backbone (Feature Extractor):
            - Block 1: Conv2d(3 → 32) + BatchNorm + LeakyReLU + MaxPool (32 → 16)
            - Block 2: Conv2d(32 → 64) + BatchNorm + LeakyReLU + MaxPool (16 → 8)
            - Block 3: Conv2d(64 → 128) + BatchNorm + LeakyReLU + MaxPool (8 → 4)

        - Classifier:
            - Flatten (128 × 4 × 4 = 2048)
            - Linear(2048 → 256) + LeakyReLU + Dropout(0.3)
            - Linear(256 → 128) + LeakyReLU

        - Head:
            - Linear(128 → 1)


        - Output:
            - Single logit for binary classification (e.g., Real vs Fake)

    Loss Function
    -------------
        - Trained using BCEWithLogitsLoss

    Notes
    -----
        - Dropout is used to reduce overfitting.
        - The architecture is lightweight and suitable for small image datasets.
    Inference
    ---------
        - Please pass the output with (torch.sigmoid(model(data)) >0.5).long()
    """

    def __init__(self, lr: float = 0.005, dropout: float = 0.1, optimizer_cfg: Any | None = None) -> None:
        super().__init__()

        self.dropout = dropout
        self.lr = lr
        self.optimizer_cfg = optimizer_cfg

        self.save_hyperparameters({"lr": lr, "dropout": dropout})

        self.backbone = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),  # 8x8 -> 4x4
        )

        # Classification layers
        # After 3 pooling layers: 128 channels * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        self.head = nn.Linear(128, 1)  # 2 classes: Real vs Fake

        self.criterium = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor [Batch, C, H, W]")
        if x.shape[1] != 3:
            raise ValueError("Expected 3 input channels (RGB)")
        return self.head(self.classifier(self.backbone(x)))

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data).squeeze(1)
        target = target.to(dtype=logits.dtype).view_as(logits)

        loss = self.criterium(logits, target)

        preds = (logits > 0).to(dtype=target.dtype)
        acc = (preds == target).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data).squeeze(1)
        target = target.to(dtype=logits.dtype).view_as(logits)

        loss = self.criterium(logits, target)
        preds = (logits > 0).to(dtype=target.dtype)
        acc = (preds == target).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_cfg is not None:
            # Local import so unit tests importing the model don't require Hydra.
            from hydra.utils import instantiate

            return instantiate(self.optimizer_cfg, params=self.parameters())
        return optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = FakeArtClassifier()

    # Test with CIFAKE image dimensions: [batch_size, channels, height, width]
    x = randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape of model: {output.shape}")
    print(f"Model output (logits): {output}")
