import os
import sys
from pathlib import Path
from typing import Annotated, Optional
import wandb
import hydra
import pytorch_lightning as pl
import typer
from dotenv import load_dotenv
from google.cloud import storage
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from torch import save
from torch.utils.data import DataLoader


from fakeartdetector.data import cifake
from fakeartdetector.helpers import configure_loguru_file, get_hydra_output_dir, resolve_path
from fakeartdetector.model import FakeArtClassifier

load_dotenv()

app = typer.Typer()

# information for the logging
# order of levels from LEAST to MOST logging
# OFF -> CRITICAL -> ERROR -> WARNING -> INFO -> DEBUG

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def download_folder_from_gcs(gcs_path, local_path):
    # gcs_path: gs://bucket/folder/
    parts = gcs_path.replace("gs://", "").split("/")
    bucket_name = parts[0]
    prefix = "/".join(parts[1:])

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading data from {gcs_path} to {local_path}...")
    for blob in blobs:
        # Avoid directories
        if blob.name.endswith("/"):
            continue
        # Check if file is in the subfolder we want
        rel_path = str(blob.name).replace(prefix, "").lstrip("/")
        if not rel_path:
            continue  # Skip if it matches prefix exactly
        dest_path = local_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest_path))
        # logger.info(f"Downloaded {blob.name}")
    logger.info("Download complete.")


def train_impl(cfg: DictConfig, print_config: bool = False) -> None:
    # set up the logging from the hydra output dir
    output_dir = get_hydra_output_dir()
    log_path = configure_loguru_file(output_dir, filename="train_hydra.log", rotation="150 MB")

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
    logger.info(
        f"lr: {hparams['lr']}, "
        f"epochs: {hparams['epochs']}, "
        f"batch_size: {hparams['batch_size']}, "
        f"precision: {hparams.get('precision', '32')}"
    )

    # Set up the LightningModule (Lightning will move it to the right device).
    model = FakeArtClassifier(lr=hparams["lr"], dropout=hparams["dropout"], optimizer_cfg=cfg.optimizer)
    logger.info(f"Loaded Model onto memory: \n{model}")

    # Handle Data Loading (GCS vs Local)
    data_path = cfg.dataset.dataset.path
    if data_path.startswith("gs://"):
        local_data_path = "data/processed"
        download_folder_from_gcs(data_path, local_data_path)
        train_set, val_set = cifake(processed_dir=local_data_path)
    else:
        train_set, val_set = cifake(processed_dir=data_path)

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

    # Set up loggers
    csv_logger = CSVLogger(save_dir=str(output_dir), name="lightning")
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")

    # NEW: W&B Logger
    wb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        save_dir=str(output_dir),
        log_model="all",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    _ = wb_logger.experiment

    loggers = [wb_logger, csv_logger, tb_logger]

    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(wb_logger.experiment.dir) / "checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="epoch{epoch}-val_loss{val_loss:.4f}",
        save_on_train_epoch_end=True,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    precision = hparams.get("precision", "32")

    # Build profiler from Hydra config
    profiler_cfg = getattr(cfg, "profiler", None)
    profiler = None
    if profiler_cfg and profiler_cfg.get("type") != "none":
        if profiler_cfg.type == "simple":
            profiler = SimpleProfiler()
        elif profiler_cfg.type == "advanced":
            profiler = AdvancedProfiler()
        elif profiler_cfg.type == "pytorch":
            # Use Hydra output dir for profiler traces
            profiler_dir = output_dir / "profiler"
            profiler_dir.mkdir(parents=True, exist_ok=True)

            # Handle schedule parameter (can be dict or None)
            schedule_cfg = profiler_cfg.get("schedule", None)
            schedule = None
            if schedule_cfg and isinstance(schedule_cfg, dict):
                from torch.profiler import schedule as torch_schedule

                schedule = torch_schedule(
                    wait=int(schedule_cfg.get("wait", 1)),
                    warmup=int(schedule_cfg.get("warmup", 1)),
                    active=int(schedule_cfg.get("active", 3)),
                    repeat=int(schedule_cfg.get("repeat", 2)),
                )

            # Set up TensorBoard trace if enabled
            on_trace_ready = None
            if profiler_cfg.get("tensorboard", False):
                from torch.profiler import tensorboard_trace_handler

                on_trace_ready = tensorboard_trace_handler(str(profiler_dir))

            profiler = PyTorchProfiler(
                dirpath=str(profiler_dir),
                filename=str(profiler_cfg.get("filename", "trace")),
                schedule=schedule,
                on_trace_ready=on_trace_ready,
                export_to_trace=bool(profiler_cfg.get("export_to_trace", True)),
                with_stack=bool(profiler_cfg.get("with_stack", False)),
                record_module_names=bool(profiler_cfg.get("record_module_names", True)),
                profile_memory=bool(profiler_cfg.get("profile_memory", True)),
            )

    trainer = pl.Trainer(
        max_epochs=int(hparams["epochs"]),
        precision=precision,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(output_dir),
        logger=loggers,
        callbacks=[checkpoint_cb, early_stopping_callback],
        log_every_n_steps=50,
        profiler=profiler,
    )

    logger.info(50 * "=")
    logger.info("Starting Lightning training")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("Training complete")

    checkpoint_dir = Path(wb_logger.experiment.dir) / "checkpoints"
    print(checkpoint_dir)

    # CREATE ARTIFACT
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        logger.info(f"Manual Artifact Trigger: Uploading {checkpoint_dir}")
        try:
            # Define the Artifact
            folder_artifact = wandb.Artifact(name=f"model-checkpoints-{wb_logger.version}", type="model")
            # Add the entire directory to the manifest
            folder_artifact.add_dir(str(checkpoint_dir))
            # Log it to the server
            artifact = wandb.log_artifact(folder_artifact)
            artifact.wait()  # Wait for upload to complete
            logger.info("Artifact logged successfully")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    else:
        logger.warning("Checkpoints folder is empty or missing locally.")
        # Mandatory: Wait for the upload to hit 100%
    wandb.finish(quiet=True)

    # Save profiler output to file
    if profiler:
        if isinstance(profiler, AdvancedProfiler):
            profiler_output = profiler.summary()
            profiler_file = output_dir / "profiler" / "advanced_profiler.txt"
            profiler_file.parent.mkdir(parents=True, exist_ok=True)
            with open(profiler_file, "w") as f:
                f.write(profiler_output)
            logger.info(f"Advanced profiler output saved to: {profiler_file}")
            logger.info("\n" + "=" * 80)
            logger.info("PROFILER SUMMARY")
            logger.info("=" * 80)
            logger.info(profiler_output)
        elif isinstance(profiler, PyTorchProfiler):
            logger.info(f"Profiler traces saved to: {output_dir / 'profiler'}")
            if profiler_cfg is not None and profiler_cfg.get("tensorboard", False):
                logger.info(f"View in TensorBoard: tensorboard --logdir={output_dir / 'profiler'}")
            else:
                logger.info("View trace.json in Chrome: chrome://tracing")

    save_path = cfg.dataset.savedTo.path
    if save_path.startswith("gs://"):
        # Local save first
        local_path = "model.pth"
        save(model.state_dict(), local_path)
        logger.info(f"Saved model locally to: {local_path}")

        # Parse bucket and blob
        # gs://bucket_name/path/to/model.pth
        try:
            parts = save_path.replace("gs://", "").split("/")
            bucket_name = parts[0]
            blob_name = "/".join(parts[1:])

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded model to GCS: {save_path}")
        except Exception as e:
            logger.error(f"Failed to upload model to GCS: {e}")

        # For compatibility with subsequent logic
        model_path = Path(local_path)
    else:
        model_path = resolve_path(cfg.dataset.savedTo.path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save(model.state_dict(), str(model_path))
        logger.info(f"Saved model to: {model_path}")

    # Log TensorBoard info
    tb_log_dir = output_dir / "tensorboard"
    logger.info(f"TensorBoard logs saved to: {tb_log_dir}")
    logger.info(f"View training metrics: tensorboard --logdir={tb_log_dir}")

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


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    lr: Annotated[Optional[float], typer.Option(help="Override learning rate")] = None,
    epochs: Annotated[Optional[int], typer.Option(help="Override epochs")] = None,
    batch_size: Annotated[Optional[int], typer.Option(help="Override batch size")] = None,
    num_workers: Annotated[Optional[int], typer.Option(help="Override number of data loading workers")] = None,
    precision: Annotated[Optional[str], typer.Option(help="Override precision (e.g. 32, 16-mixed, bf16-mixed)")] = None,
    profiler: Annotated[
        Optional[str], typer.Option(help="Choose profiler config (none|simple|advanced|pytorch)")
    ] = None,
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
    if profiler is not None:
        overrides.append(f"profiler={profiler}")

    # Add any extra arguments passed to the CLI (e.g. dataset.savedTo.path=...)
    if ctx.args:
        overrides.extend(ctx.args)

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
