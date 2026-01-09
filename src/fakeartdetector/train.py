import os
from pathlib import Path
from typing import Annotated

import hydra
import matplotlib.pyplot as plt
import typer
from loguru import logger
from omegaconf import OmegaConf
from torch import cuda, device, manual_seed, optim, save
from torch.backends import mps
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.model import FakeArtClassifier

app = typer.Typer()
DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")

# information for the logging
# order of levels from LEAST to MOST logging
# OFF -> CRITICAL -> ERROR -> WARNING -> INFO -> DEBUG

CONFIG_DIR = "../../configs"
# CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


# def train(lr: float = 1e-3, epochs: int = 5, batch_size: int = 32) -> None:
# @hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="default_config.yaml")
@app.command()
def train(
    lr: Annotated[float, typer.Option(help="Override learning rate")] = None,
    epochs: Annotated[int, typer.Option(help="Override epochs")] = None,
    batch_size: Annotated[int, typer.Option(help="Override batch size")] = None,
    config_name: Annotated[str, typer.Option(help="Config file name")] = "default_config.yaml",
    print_config: Annotated[bool, typer.Option(help="Print config")] = False,
    experiment: Annotated[str, typer.Option(help="Choose experimental config")] = "exp1",
) -> None:
    """
    Train model using Hydra configuration + CLI overrides.
    """
    with hydra.initialize(version_base=None, config_path=str(CONFIG_DIR)):
        cfg = hydra.compose(config_name=config_name, overrides=[f"experiment={experiment}"])

    if lr is not None:
        cfg.experiment.hyperparameters.lr = lr
    if epochs is not None:
        cfg.experiment.hyperparameters.epochs = epochs
    if batch_size is not None:
        cfg.experiment.hyperparameters.batch_size = batch_size

    # the line bellow only works with the @hydra.main decorator
    # hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    hydra_path = Path(cfg.experiment.logging.path)
    logger.add(os.path.join(hydra_path, "train_logger.log"), rotation="150 MB")

    logger.info(f"Loading model with configuration: \n {OmegaConf.to_yaml(cfg)}")

    hparams = cfg.experiment.hyperparameters
    manual_seed(hparams["seed"])

    logger.info("Training day and night")
    logger.info(f"lr: {hparams['lr']}, epochs: {hparams['epochs']}, batch_size: {hparams['batch_size']}")

    model = FakeArtClassifier(lr=hparams["lr"], dropOut=hparams["dropOut"]).to(DEVICE)
    logger.info(f"Loaded Model onto memory: \n{model}")

    train_set, _ = cifake()
    logger.info("Loaded training set")

    train_loader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    logger.debug("Loaded train_loader No problems")

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    model.train()
    for epoch in range(hparams["epochs"]):
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
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    logger.info("Training complete")
    save(model.state_dict(), "models/model.pth")
    logger.info("Saved model")

    # make a nice statistic plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    fig.savefig("reports/figures/training_statistics.png")
    logger.info("Saved training stats figure")


def main():
    app()


if __name__ == "__main__":
    main()
