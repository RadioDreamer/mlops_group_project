import sys
from pathlib import Path
from typing import Annotated

import hydra
import matplotlib.pyplot as plt
import typer
from hydra.utils import instantiate
from loguru import logger
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import cuda, device, manual_seed, save
from torch.backends import mps
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.model import FakeArtClassifier

app = typer.Typer()
DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")

# information for the logging
# order of levels from LEAST to MOST logging
# OFF -> CRITICAL -> ERROR -> WARNING -> INFO -> DEBUG

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _configure_logging(output_dir: Path) -> Path:
    """Configure Loguru to write into Hydra's per-run output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_hydra.log"

    # Prevent duplicate handlers if train_impl is invoked multiple times.
    logger.remove()
    logger.add(str(log_path), rotation="150 MB")
    return log_path


def _resolve_save_path(path_str: str) -> Path:
    """Resolve a possibly relative path against the current working directory."""
    path = Path(path_str)
    return path if path.is_absolute() else (Path.cwd() / path)


def train_impl(cfg: DictConfig, print_config: bool = False) -> None:
    # set up the logging from the hydra output dir
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception as exc:
        raise RuntimeError(
            "Hydra is not initialized. Run via the hydra entrypoint `train_hydra()` or the Typer command `train`, "
            "not by calling `train_impl()` directly."
        ) from exc
    log_path = _configure_logging(output_dir)

    # we can print the config if we want to
    if print_config:
        typer.echo(OmegaConf.to_yaml(cfg))

    # here we put the config into the logs
    logger.info(f"Hydra output_dir: {output_dir}")
    logger.info(f"Training log_path: {log_path}")
    logger.info(f"Loading model with configuration: \n {OmegaConf.to_yaml(cfg)}")

    hparams = cfg.experiment.hyperparameters
    manual_seed(hparams["seed"])

    logger.info("Training day and night")
    logger.info(f"lr: {hparams['lr']}, epochs: {hparams['epochs']}, batch_size: {hparams['batch_size']}")

    # here we set up the model, data, optimizer, etc.
    model = FakeArtClassifier(lr=hparams["lr"], dropout=hparams["dropout"]).to(DEVICE)
    logger.info(f"Loaded Model onto memory: \n{model}")

    train_set, _ = cifake()
    logger.info("Loaded training set")

    train_loader = DataLoader(train_set, batch_size=hparams["batch_size"], shuffle=True)
    logger.debug("Loaded train_loader No problems")

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Training loop
    statistics = {"train_loss": [], "train_accuracy": []}
    model.train()
    logger.info(50 * "=")
    logger.info("Strarting training")
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

    logger.info(50 * "=")
    logger.info("Training complete")
    model_path = _resolve_save_path(cfg.dataset.savedTo.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save(model.state_dict(), str(model_path))
    logger.info(f"Saved model to: {model_path}")

    # make a nice statistic plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    fig_dir = Path("reports") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "training_statistics.png"
    fig.savefig(str(fig_path))
    logger.info(f"Saved training stats figure to: {fig_path}")


@app.command()
def train(
    lr: Annotated[float, typer.Option(help="Override learning rate")] = None,
    epochs: Annotated[int, typer.Option(help="Override epochs")] = None,
    batch_size: Annotated[int, typer.Option(help="Override batch size")] = None,
    config_name: Annotated[str, typer.Option(help="Config file name")] = "default_config.yaml",
    print_config: Annotated[bool, typer.Option(help="Print config")] = False,
    experiment: Annotated[str, typer.Option(help="Choose experimental config")] = "base",
    dataset: Annotated[str, typer.Option(help="Choose dataset config")] = "base",
    logging: Annotated[str, typer.Option(help="Choose logging config")] = "base",
    optimizer: Annotated[str, typer.Option(help="Choose optimizer config (e.g. adam, sgd)")] = "adam",
) -> None:
    """Typer wrapper that forwards options as Hydra overrides."""
    overrides: list[str] = [
        f"experiment={experiment}",
        f"dataset={dataset}",
        f"logging={logging}",
        f"optimizer={optimizer}",
    ]
    if lr is not None:
        overrides.append(f"experiment.hyperparameters.lr={lr}")
    if epochs is not None:
        overrides.append(f"experiment.hyperparameters.epochs={epochs}")
    if batch_size is not None:
        overrides.append(f"experiment.hyperparameters.batch_size={batch_size}")

    # Build a Hydra-decorated runner so Hydra (not Typer) owns config parsing.
    decorated = hydra.main(
        version_base="1.3",
        config_path=str(CONFIG_DIR),
        config_name=config_name,
    )(lambda cfg: train_impl(cfg, print_config=print_config))

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *overrides]
        decorated()
    finally:
        sys.argv = original_argv


def main():
    """runs the app"""
    app()


if __name__ == "__main__":
    main()
