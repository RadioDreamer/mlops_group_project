import matplotlib.pyplot as plt
import torch
import typer
from numpy import unique
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from fakeartdetector.model import FakeArtClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model = FakeArtClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    embeddings, targets = [], []

    with torch.inference_mode():
        for batch in loader:
            images, target = batch
            predictions = model.classifier(model.backbone(images.to(DEVICE)))
            embeddings.append(predictions.cpu())
            targets.append(target)
    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in unique(targets):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=f"Class {i}", alpha=0.6)
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)
