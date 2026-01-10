import sys
from pathlib import Path
from typing import Annotated

import hydra
import typer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import cuda, device, load, no_grad, sigmoid
from torch.backends import mps
from torch.utils.data import DataLoader

from fakeartdetector.data import cifake
from fakeartdetector.helpers import configure_loguru_file, get_hydra_output_dir, resolve_path
from fakeartdetector.model import FakeArtClassifier

app = typer.Typer()
DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def evaluate_checkpoint(model_checkpoint: str, batch_size: int = 32, threshold: float = 0.5) -> float:
    """Evaluate a trained model checkpoint and return accuracy."""
    logger.info("Evaluating model")
    logger.info(f"checkpoint: {model_checkpoint}")
    logger.info(f"batch_size: {batch_size}, threshold: {threshold}")

    model = FakeArtClassifier().to(DEVICE)
    state_dict = load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    _, test_set = cifake()
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    correct, total = 0, 0
    with no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = (sigmoid(model(img).squeeze(1)) > threshold).long()
            correct += (y_pred == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    logger.info(f"Test accuracy: {accuracy}")
    return accuracy


def evaluate_impl(cfg: DictConfig, print_config: bool = False) -> None:
    output_dir = get_hydra_output_dir()
    log_path = configure_loguru_file(output_dir, filename="evaluate.log", rotation="50 MB")

    if print_config:
        typer.echo(OmegaConf.to_yaml(cfg))

    ckpt = resolve_path(cfg.evaluate.model_checkpoint)
    logger.info(f"Hydra output_dir: {output_dir}")
    logger.info(f"Eval log_path: {log_path}")
    logger.info(f"Resolved checkpoint: {ckpt}")

    accuracy = evaluate_checkpoint(
        model_checkpoint=str(ckpt),
        batch_size=int(cfg.evaluate.batch_size),
        threshold=float(cfg.evaluate.threshold),
    )
    typer.echo(f"Test accuracy: {accuracy}")


@app.command()
def evaluate(
    model_checkpoint: Annotated[str, typer.Option(help="Override checkpoint path")] = None,
    batch_size: Annotated[int, typer.Option(help="Override batch size")] = None,
    threshold: Annotated[float, typer.Option(help="Override sigmoid threshold")] = None,
    config_name: Annotated[str, typer.Option(help="Config file name")] = "default_config.yaml",
    print_config: Annotated[bool, typer.Option(help="Print resolved config")] = False,
    dataset: Annotated[str, typer.Option(help="Choose dataset config group")] = "base",
    evaluate_cfg: Annotated[str, typer.Option(help="Choose evaluate config group")] = "base",
) -> None:
    """Evaluate using Hydra configuration + optional CLI overrides."""
    overrides: list[str] = [
        f"dataset={dataset}",
        f"evaluate={evaluate_cfg}",
    ]
    if model_checkpoint is not None:
        overrides.append(f"evaluate.model_checkpoint={model_checkpoint}")
    if batch_size is not None:
        overrides.append(f"evaluate.batch_size={batch_size}")
    if threshold is not None:
        overrides.append(f"evaluate.threshold={threshold}")

    decorated = hydra.main(
        version_base="1.3",
        config_path=str(CONFIG_DIR),
        config_name=config_name,
    )(lambda cfg: evaluate_impl(cfg, print_config=print_config))

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *overrides]
        decorated()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    app()
