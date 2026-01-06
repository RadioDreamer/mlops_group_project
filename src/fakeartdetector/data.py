from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from datasets import load_dataset
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision import transforms

app = typer.Typer()


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Standard normalization for image tensors."""
    # Using per-channel mean/std is standard for RGB
    return (images - images.mean(dim=(0, 2, 3), keepdim=True)) / images.std(dim=(0, 2, 3), keepdim=True)


@app.command()
def preprocess_data(processed_dir: str = "data/processed") -> None:
    """
    Downloads CIFAKE from Hugging Face, transforms to tensors,
    and saves to processed_dir for DVC tracking.
    """
    path = Path(processed_dir)
    path.mkdir(parents=True, exist_ok=True)

    typer.echo("Loading dataset from Hugging Face...")
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")

    # Define the transform: Resize (just in case), ToTensor (scales to 0-1)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    for split in ["train", "test"]:
        typer.echo(f"Processing {split} split...")
        images = []
        labels = []

        for item in dataset[split]:
            # item["image"] is a PIL object
            img = transform(item["image"].convert("RGB"))
            images.append(img)
            labels.append(item["label"])

        # Stack into [N, 3, 32, 32]
        img_tensor = torch.stack(images)
        label_tensor = torch.tensor(labels).long()

        # Save tensors
        torch.save(img_tensor, path / f"{split}_images.pt")
        torch.save(label_tensor, path / f"{split}_target.pt")

    typer.echo(f"Finished! Data saved in {processed_dir}")


def cifake() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test dataloaders for CIFAKE."""
    train_images = torch.load("data/processed/train_images.pt", weights_only=True)
    train_target = torch.load("data/processed/train_target.pt", weights_only=True)
    test_images = torch.load("data/processed/test_images.pt", weights_only=True)
    test_target = torch.load("data/processed/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot RGB images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)

    for ax, im, label in zip(grid, images, target):
        # Permute from [C, H, W] to [H, W, C] for matplotlib
        ax.imshow(im.permute(1, 2, 0))
        ax.set_title(f"Label: {'Fake' if label.item() == 1 else 'Real'}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    app()
