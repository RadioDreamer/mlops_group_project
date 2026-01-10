from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from numpy import unique
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from fakeartdetector.model import FakeArtClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


app = typer.Typer(add_completion=False)


@app.command()
def visualize(
    model_checkpoint: str = typer.Argument(..., help="Path to a saved model checkpoint (.pth)"),
    figure_name: str = typer.Option("embeddings.png", help="Output figure file name"),
    output_dir: str = typer.Option("reports/figures", help="Directory to write the figure into"),
    data_dir: str = typer.Option("data/processed", help="Directory containing test_images.pt and test_target.pt"),
    batch_size: int = typer.Option(32, help="Batch size for embedding extraction"),
    pca_threshold_dim: int = typer.Option(500, help="Apply PCA if embedding dimensionality is above this threshold"),
    pca_n_components: int = typer.Option(100, help="Number of PCA components (if PCA is applied)"),
    tsne_perplexity: float = typer.Option(30.0, help="t-SNE perplexity"),
    tsne_learning_rate: str = typer.Option("auto", help="t-SNE learning rate (float as string, or 'auto')"),
    seed: int = typer.Option(42, help="Random seed for dimensionality reduction"),
) -> None:
    """Visualize model embeddings with t-SNE and save a scatter plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = FakeArtClassifier().to(DEVICE)
    state_dict = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    processed = Path(data_dir)
    test_images = torch.load(processed / "test_images.pt", map_location="cpu")
    test_target = torch.load(processed / "test_target.pt", map_location="cpu")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    embeddings, targets = [], []

    with torch.inference_mode():
        for batch in loader:
            images, target = batch
            predictions = model.classifier(model.backbone(images.to(DEVICE)))
            embeddings.append(predictions.cpu())
            targets.append(target)
    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > pca_threshold_dim:
        pca = PCA(n_components=min(pca_n_components, embeddings.shape[1]), random_state=seed)
        embeddings = pca.fit_transform(embeddings)
    learning_rate: float | str
    if tsne_learning_rate.strip().lower() == "auto":
        learning_rate = "auto"
    else:
        try:
            learning_rate = float(tsne_learning_rate)
        except ValueError as exc:
            raise typer.BadParameter("tsne-learning-rate must be a float or 'auto'") from exc

    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate=learning_rate,
        random_state=seed,
        init="pca",
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in unique(targets):
        mask = targets == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f"Class {i}", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / figure_name)


if __name__ == "__main__":
    app()
