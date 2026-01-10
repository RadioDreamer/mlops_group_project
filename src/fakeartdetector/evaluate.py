import typer
from torch import cuda, device, load, no_grad, sigmoid
from torch.backends import mps
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.model import FakeArtClassifier

DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = FakeArtClassifier().to(DEVICE)
    state_dict = load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    _, test_set = cifake()
    test_dataloader = DataLoader(test_set, batch_size=32)

    correct, total = 0, 0
    with no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = (sigmoid(model(img).squeeze(1)) > 0.5).long()
            correct += (y_pred == target).sum().item()
            total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
