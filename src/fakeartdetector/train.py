from typing import Annotated

import hydra
import matplotlib.pyplot as plt
import typer
from omegaconf import OmegaConf
from torch import cuda, device, optim, save
from torch.backends import mps
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.model import FakeArtClassifier

app = typer.Typer()
DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")


@app.command()
def train(
    lr: Annotated[float, typer.Option(help="Override learning rate")] = None,
    epochs: Annotated[int, typer.Option(help="Override epochs")] = None,
    batch_size: Annotated[int, typer.Option(help="Override batch size")] = None,
    config_name: Annotated[str, typer.Option(help="Config file name")] = "config",
    print_config: Annotated[bool, typer.Option(help="Print config")] = False,
) -> None:
    """
    Train model using Hydra configuration + CLI overrides.
    """

    with hydra.initialize(version_base=None, config_path="../../experiments"):
        cfg = hydra.compose(config_name=config_name)

    if lr is not None:
        cfg.hyperparameters.lr = lr
    if epochs is not None:
        cfg.hyperparameters.epochs = epochs
    if batch_size is not None:
        cfg.hyperparameters.batch_size = batch_size

    # Debugging: Show what we are running with
    if print_config:
        print(OmegaConf.to_yaml(cfg))

    print("Training day and night")

    # 5. Use 'cfg' to access values (dot notation)
    hparams = cfg.hyperparameters
    print(f"Final Config: lr={hparams.lr}, epochs={hparams.epochs}, batch={hparams.batch_size}")

    model = FakeArtClassifier().to(DEVICE)
    train_set, _ = cifake()

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=hparams.lr)

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(hparams.epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            logits = model(img).squeeze(1)
            loss = model.criterium(logits, target.float())

            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())

            preds = (logits > 0).long()
            accuracy = (preds == target.long()).float().mean().item()

            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    save(model.state_dict(), "models/model.pth")

    # make a nice statistic plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


def main():
    app()


if __name__ == "__main__":
    main()
