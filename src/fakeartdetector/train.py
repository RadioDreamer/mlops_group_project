import sys
import os
from pathlib import Path
from typing import Annotated

import hydra
import pytorch_lightning as pl
import typer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch import save
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.helpers import configure_loguru_file, get_hydra_output_dir, resolve_path
from fakeartdetector.model import FakeArtClassifier

app = typer.Typer()

# information for the logging
# order of levels from LEAST to MOST logging
# OFF -> CRITICAL -> ERROR -> WARNING -> INFO -> DEBUG

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def train_impl(cfg: DictConfig, print_config: bool = False) -> None:
    # set up the logging from the hydra output dir
    output_dir = get_hydra_output_dir()
    log_path = configure_loguru_file(output_dir, filename="train_hydra.log", rotation="150 MB")
    model_checkpoint_path = cfg.evaluate.model_checkpoint

    # we can print the config if we want to
    if print_config:
        typer.echo(OmegaConf.to_yaml(cfg))

    # here we put the config into the logs
    logger.info(f"Hydra output_dir: {output_dir}")
    logger.info(f"Training log_path: {log_path}")
    logger.info(f"Loading model with configuration: \n {OmegaConf.to_yaml(cfg)}")

    hparams = cfg.experiment.hyperparameters
    pl.seed_everything(int(hparams["seed"]), workers=True)

    logger.info("Training day and night")
    logger.info(f"lr: {hparams['lr']}, epochs: {hparams['epochs']}, batch_size: {hparams['batch_size']}, precision: {hparams.get('precision', '32')}")

    # Set up the LightningModule (Lightning will move it to the right device).
    model = FakeArtClassifier(lr=hparams["lr"], dropout=hparams["dropout"], optimizer_cfg=cfg.optimizer)
    logger.info(f"Loaded Model onto memory: \n{model}")

    train_set, val_set = cifake(processed_dir=cfg.dataset.dataset.path)
    logger.info("Loaded training set")

    num_workers_cfg = hparams.get("num_workers", None)
    if num_workers_cfg is None:
        cpu_count = os.cpu_count() or 0
        num_workers = min(7, max(0, cpu_count - 1))
    else:
        num_workers = int(num_workers_cfg)

    train_loader = DataLoader(
        train_set,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.debug("Loaded train/val dataloaders")

    csv_logger = CSVLogger(save_dir=str(output_dir), name="lightning")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="epoch{epoch}-val_loss{val_loss:.4f}",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")


    precision = hparams.get("precision", "32")
    
    trainer = pl.Trainer(
        max_epochs=int(hparams["epochs"]),
        precision=precision,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_dir),
        logger=csv_logger,
        callbacks=[checkpoint_cb, early_stopping_callback],
        log_every_n_steps=50,
    )

    logger.info(50 * "=")
    logger.info("Starting Lightning training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training complete")

    model_path = resolve_path(cfg.dataset.savedTo.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save(model.state_dict(), str(model_path))
    logger.info(f"Saved model to: {model_path}")

    # Save full composed Hydra config for reproducibility
    full_cfg_path = output_dir / "config_full.yaml"
    OmegaConf.save(config=cfg, f=str(full_cfg_path))
    logger.info(f"Saved full config to: {full_cfg_path}")

    # Save an artifacts manifest with key file paths
    artifacts = {
        "final_model_state_path": str(model_path),
        "best_checkpoint_path": str(checkpoint_cb.best_model_path),
        "train_log_path": str(log_path),
        "csv_logger_dir": getattr(csv_logger, "log_dir", str(Path(csv_logger.save_dir) / csv_logger.name)),
        "hydra_output_dir": str(output_dir),
        "num_workers": num_workers,
    }
    artifacts_path = output_dir / "artifacts.yaml"
    OmegaConf.save(config=OmegaConf.create(artifacts), f=str(artifacts_path))
    logger.info(f"Saved artifacts manifest to: {artifacts_path}")


@app.command()
def train(
    lr: Annotated[float, typer.Option(help="Override learning rate")] = None,
    epochs: Annotated[int, typer.Option(help="Override epochs")] = None,
    batch_size: Annotated[int, typer.Option(help="Override batch size")] = None,
    num_workers: Annotated[int, typer.Option(help="Override number of data loading workers")] = None,
    precision: Annotated[str, typer.Option(help="Override precision (e.g. 32, 16-mixed, bf16-mixed)")] = None,
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
    if num_workers is not None:
        overrides.append(f"experiment.hyperparameters.num_workers={num_workers}")
    if precision is not None:
        overrides.append(f"experiment.hyperparameters.precision={precision}")

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
