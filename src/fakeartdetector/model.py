import torch
from torch import nn


class Model(nn.Module):
    """CNN for classifying AI-generated vs human-created art (CIFAKE dataset)"""

    def __init__(self):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 8x8 -> 4x4
        )

        # Classification layers
        # After 3 pooling layers: 128 channels * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),  # 2 classes: Real vs Fake
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = Model()
    # Test with CIFAKE image dimensions: [batch_size, channels, height, width]
    x = torch.rand(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape of model: {output.shape}")
    print(f"Model output (logits): {output}")
