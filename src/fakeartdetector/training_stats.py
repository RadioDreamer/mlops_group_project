"""Load and visualize training artifacts and statistics."""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf


def load_artifacts(output_dir: str | Path) -> dict:
    """Load artifacts manifest from a training run.

    Args:
        output_dir: Path to Hydra output directory (e.g., outputs/2026-01-10/14-31-25)

    Returns:
        Dictionary with paths to model, checkpoint, logs, etc.
    """
    output_dir = Path(output_dir)
    artifacts_path = output_dir / "artifacts.yaml"

    if not artifacts_path.exists():
        raise FileNotFoundError(f"artifacts.yaml not found at {artifacts_path}")

    artifacts = OmegaConf.load(str(artifacts_path))
    return OmegaConf.to_container(artifacts, resolve=True)


def load_config(output_dir: str | Path) -> dict:
    """Load the full composed Hydra config from a training run.

    Args:
        output_dir: Path to Hydra output directory

    Returns:
        Full OmegaConf config object
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "config_full.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config_full.yaml not found at {config_path}")

    return OmegaConf.load(str(config_path))


def load_metrics(output_dir: str | Path) -> pd.DataFrame:
    """Load training metrics from CSV logger.

    Args:
        output_dir: Path to Hydra output directory

    Returns:
        DataFrame with epoch, train_loss, train_acc, val_loss, val_acc, etc.
    """
    output_dir = Path(output_dir)

    # Find the metrics CSV file (usually in lightning/version_0/metrics.csv)
    csv_files = list(output_dir.glob("lightning/version_*/metrics.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No metrics.csv found in {output_dir}")

    metrics_path = csv_files[0]
    df = pd.read_csv(metrics_path)

    return df


def print_training_summary(output_dir: str | Path) -> None:
    """Print a nice summary of a training run.

    Args:
        output_dir: Path to Hydra output directory
    """
    output_dir = Path(output_dir)

    print(f"\n{'=' * 60}")
    print(f"Training Summary: {output_dir.name}")
    print(f"{'=' * 60}\n")

    # Load and display config
    config = load_config(output_dir)
    hparams = config.experiment.hyperparameters
    print("Hyperparameters:")
    print(f"  lr: {hparams.lr}")
    print(f"  epochs: {hparams.epochs}")
    print(f"  batch_size: {hparams.batch_size}")
    print(f"  precision: {hparams.get('precision', '32')}")
    print(f"  dropout: {hparams.dropout}\n")

    # Load and display metrics
    metrics = load_metrics(output_dir)
    print("Final Metrics:")
    if "train_loss" in metrics.columns:
        final_train_loss = metrics["train_loss"].dropna().iloc[-1]
        print(f"  Final train_loss: {final_train_loss:.4f}")
    if "train_acc" in metrics.columns:
        final_train_acc = metrics["train_acc"].dropna().iloc[-1]
        print(f"  Final train_acc: {final_train_acc:.4f}")
    if "val_loss" in metrics.columns:
        final_val_loss = metrics["val_loss"].dropna().iloc[-1]
        print(f"  Final val_loss: {final_val_loss:.4f}")
    if "val_acc" in metrics.columns:
        final_val_acc = metrics["val_acc"].dropna().iloc[-1]
        print(f"  Final val_acc: {final_val_acc:.4f}")

    print()

    # Load and display artifacts
    artifacts = load_artifacts(output_dir)
    print("Saved Artifacts:")
    print(f"  Model: {artifacts['final_model_state_path']}")
    print(f"  Best checkpoint: {artifacts['best_checkpoint_path']}")
    print(f"  Full config: {output_dir / 'config_full.yaml'}")
    print(f"  Artifacts manifest: {output_dir / 'artifacts.yaml'}")
    print(f"  CSV metrics: {output_dir / 'lightning/version_0/metrics.csv'}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m fakeartdetector.training_stats <output_dir>")
        print("Example: python -m fakeartdetector.training_stats outputs/2026-01-10/14-31-25")
        sys.exit(1)

    output_dir = sys.argv[1]
    print_training_summary(output_dir)
