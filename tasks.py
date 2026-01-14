import os
from pathlib import Path

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "fakeartdetector"
PYTHON_VERSION = "3.11"

# Valid precision options (MPS-compatible for macOS)
# Note: 16-mixed and 64-bit precisions don't work with MPS
VALID_PRECISIONS = ["32", "bf16-mixed", "bf16", "bf16-true"]


def _run(ctx: Context, command: str) -> None:
    ctx.run(command, echo=True, pty=not WINDOWS)


def _list_yaml_stems(dir_path: Path) -> list[str]:
    if not dir_path.exists():
        return []
    return sorted([p.stem for p in dir_path.glob("*.yaml") if p.is_file()])


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    _run(ctx, f"uv run src/{PROJECT_NAME}/data.py data/processed")


@task
def startapi(ctx: Context) -> None:
    """Starts api using uvicorn"""
    _run(ctx, "uv run uvicorn --reload --app-dir src/fakeartdetector/ api:app")


@task(
    help={
        "lr": "Override learning rate (float)",
        "epochs": "Override epochs (int)",
        "batch_size": "Override batch size (int)",
        "num_workers": "Override number of data loading workers (0 recommended for macOS)",
        "precision": f"Override precision: {', '.join(VALID_PRECISIONS)}",
        "profiler": "Profiler config group name (none|simple|advanced|pytorch)",
        "experiment": "Experiment config group name (e.g. base)",
        "dataset": "Dataset config group name (e.g. base)",
        "logging": "Logging config group name (e.g. base)",
        "optimizer": "Optimizer config group name (e.g. adam, sgd)",
        "config_name": "Top-level config file (default_config.yaml)",
        "print_config": "Print resolved Hydra config before training",
    }
)
def train(
    ctx,
    lr: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    profiler: str | None = None,
    experiment: str | None = None,
    dataset: str | None = None,
    logging: str | None = None,
    optimizer: str | None = None,
    config_name: str = "default_config.yaml",
    print_config: bool = False,
):
    """
    Train using the Hydra-backed Typer CLI in `fakeartdetector.train`.

    Examples:
      - uv run invoke train
      - uv run invoke train --epochs 2 --lr 0.001
      - uv run invoke train --experiment trials1
    """
    # Validate precision if provided
    if precision is not None and precision not in VALID_PRECISIONS:
        print(f"Error: Invalid precision '{precision}'")
        print(f"Valid options: {', '.join(VALID_PRECISIONS)}")
        return

    cmd = f"uv run python -m {PROJECT_NAME}.train"

    # Append flags only if provided (Typer/Hydra will handle the rest)
    if lr is not None:
        cmd += f" --lr {lr}"
    if epochs is not None:
        cmd += f" --epochs {epochs}"
    if batch_size is not None:
        cmd += f" --batch-size {batch_size}"
    if num_workers is not None:
        cmd += f" --num-workers {num_workers}"
    if precision is not None:
        cmd += f" --precision {precision}"
    if profiler is not None:
        cmd += f" --profiler {profiler}"
    if experiment is not None:
        cmd += f" --experiment {experiment}"
    if dataset is not None:
        cmd += f" --dataset {dataset}"
    if logging is not None:
        cmd += f" --logging {logging}"
    if optimizer is not None:
        cmd += f" --optimizer {optimizer}"
    if config_name:
        cmd += f" --config-name {config_name}"
    if print_config:
        cmd += " --print-config"

    _run(ctx, cmd)


@task
def train_help(ctx: Context) -> None:
    """Show the training CLI help (Typer options)."""
    _run(ctx, f"uv run python -m {PROJECT_NAME}.train --help")


@task
def list_configs(ctx: Context) -> None:
    """List available Hydra config group options (experiment/dataset/logging)."""
    configs_dir = Path(__file__).resolve().parent / "configs"
    experiments = _list_yaml_stems(configs_dir / "experiment")
    datasets = _list_yaml_stems(configs_dir / "dataset")
    loggings = _list_yaml_stems(configs_dir / "logging")
    optimizers = _list_yaml_stems(configs_dir / "optimizer")
    profilers = _list_yaml_stems(configs_dir / "profiler")

    print("Available config options:")
    print(f"  experiment: {', '.join(experiments) if experiments else '(none found)'}")
    print(f"  dataset:    {', '.join(datasets) if datasets else '(none found)'}")
    print(f"  logging:    {', '.join(loggings) if loggings else '(none found)'}")
    print(f"  optimizer:  {', '.join(optimizers) if optimizers else '(none found)'}")
    print(f"  profiler:   {', '.join(profilers) if profilers else '(none found)'}")


@task
def list_precisions(ctx: Context) -> None:
    """List available precision options for training (MPS-compatible)."""
    precisions = {
        "32": "Full precision (32-bit floats) - safe, slower",
        "bf16-mixed": "Mixed precision with bfloat16 (recommended for MPS)",
        "bf16": "Brain float 16-bit",
        "bf16-true": "Brain float 16-bit, explicit",
    }
    print("Available precision options (MPS-compatible for macOS):")
    for precision, desc in sorted(precisions.items()):
        print(f"  {precision:<12} {desc}")


@task(
    help={
        "logdir": "Path to TensorBoard logs directory (default: outputs/)",
        "port": "Port to run TensorBoard on (default: 6006)",
    }
)
def tensorboard(ctx: Context, logdir: str = "outputs", port: int = 6006) -> None:
    """Start TensorBoard to view training metrics and profiler traces.

    Examples:
        uv run invoke tensorboard
        uv run invoke tensorboard --logdir outputs/2026-01-10/14-31-25/tensorboard
        uv run invoke tensorboard --port 6007
    """
    print(f"Starting TensorBoard on http://localhost:{port}")
    print(f"Viewing logs from: {logdir}")
    _run(ctx, f"uv run tensorboard --logdir={logdir} --port={port}")


@task
def test(ctx: Context) -> None:
    """Run tests."""
    _run(ctx, "uv run coverage run -m pytest tests/")
    _run(ctx, "uv run coverage report -m -i")


@task(
    help={
        "checkpoint": "Override checkpoint path (defaults to cfg.evaluate.model_checkpoint)",
        "batch_size": "Override batch size",
        "threshold": "Override sigmoid threshold",
        "config_name": "Top-level config file (default_config.yaml)",
        "dataset": "Dataset config group name (e.g. base)",
        "evaluate_cfg": "Evaluate config group name (e.g. base)",
        "print_config": "Print resolved Hydra config before evaluating",
    }
)
def evaluate(
    ctx: Context,
    checkpoint: str | None = None,
    batch_size: int | None = None,
    threshold: float | None = None,
    config_name: str = "default_config.yaml",
    dataset: str = "base",
    evaluate_cfg: str = "base",
    print_config: bool = False,
) -> None:
    """Evaluate using the Hydra-backed CLI in `fakeartdetector.evaluate`."""
    cmd = f"uv run python -m {PROJECT_NAME}.evaluate"
    cmd += f" --config-name {config_name}"
    cmd += f" --dataset {dataset}"
    cmd += f" --evaluate-cfg {evaluate_cfg}"
    if checkpoint is not None:
        cmd += f" --model-checkpoint {checkpoint}"
    if batch_size is not None:
        cmd += f" --batch-size {batch_size}"
    if threshold is not None:
        cmd += f" --threshold {threshold}"
    if print_config:
        cmd += " --print-config"

    _run(ctx, cmd)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    _run(ctx, f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}")
    _run(ctx, f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}")


# data version commands
@task
def dvc(ctx, folder="data", message="Add new data"):
    ctx.run(f"dvc add {folder}")
    ctx.run(f"git add {folder}.dvc .gitignore")
    ctx.run(f"git commit -m '{message}'")
    ctx.run("git push")
    ctx.run("dvc push")


@task
def pull_data(ctx):
    ctx.run("dvc pull")


# simple visualization command
@task(
    help={
        "checkpoint": "Path to the saved model checkpoint (.pth)",
        "figure_name": "Output figure file name (e.g. embeddings.png)",
        "output_dir": "Directory to write the figure into",
        "data_dir": "Directory containing test_images.pt and test_target.pt",
        "batch_size": "Batch size for embedding extraction",
        "pca_threshold_dim": "Apply PCA if embedding dimensionality is above this threshold",
        "pca_n_components": "Number of PCA components (if PCA is applied)",
        "tsne_perplexity": "t-SNE perplexity",
        "tsne_learning_rate": "t-SNE learning rate (float as string or 'auto')",
        "seed": "Random seed for dimensionality reduction",
    }
)
def visualize(
    ctx: Context,
    checkpoint: str = "models/base_model.pth",
    figure_name: str = "embeddings.png",
    output_dir: str = "reports/figures",
    data_dir: str = "data/processed",
    batch_size: int = 32,
    pca_threshold_dim: int = 500,
    pca_n_components: int = 100,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str = "auto",
    seed: int = 42,
) -> None:
    """Visualize embeddings using the Typer CLI in `fakeartdetector.visualize`."""
    cmd = f"uv run python -m {PROJECT_NAME}.visualize {checkpoint}"
    cmd += f" --figure-name {figure_name}"
    cmd += f" --output-dir {output_dir}"
    cmd += f" --data-dir {data_dir}"
    cmd += f" --batch-size {batch_size}"
    cmd += f" --pca-threshold-dim {pca_threshold_dim}"
    cmd += f" --pca-n-components {pca_n_components}"
    cmd += f" --tsne-perplexity {tsne_perplexity}"
    cmd += f" --tsne-learning-rate {tsne_learning_rate}"
    cmd += f" --seed {seed}"

    _run(ctx, cmd)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    _run(ctx, "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build")


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    _run(ctx, "uv run mkdocs serve --config-file docs/mkdocs.yaml")


@task
def make_req_txt(ctx: Context) -> None:
    """Export dependencies to requirements.txt and requirements_dev.txt.

    - requirements.txt: runtime dependencies (default group)
    - requirements_dev.txt: runtime + dev dependencies
    """
    _run(ctx, "uv export --format requirements-txt --no-hashes -o requirements.txt")
    _run(ctx, "uv export --format requirements-txt --no-hashes --group dev -o requirements_dev.txt")


# Git commands
@task(help={"message": "The commit message"})
def push(ctx: Context, message: str = "chore: empty commit") -> None:
    """
    Add all changes, commit, and push.
    """
    ctx.run("git add .")
    ctx.run(f'git commit -m "{message}"')
    branch = ctx.run("git rev-parse --abbrev-ref HEAD", hide=True).stdout.strip()
    result = ctx.run(f"git push origin {branch}", warn=True)
    if result.failed:
        print(f"No upstream set for {branch}. Setting upstream...")
        ctx.run(f"git push --set-upstream origin {branch}")
