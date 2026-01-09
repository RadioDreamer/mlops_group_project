import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "fakeartdetector"
PYTHON_VERSION = "3.11"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/processed", echo=True, pty=not WINDOWS)


@task(help={"lr": "Override learning rate", "epochs": "Number of epochs"})
def train(c, lr=None, epochs=None):
    """
    Run the training script using 'uv run' for environment isolation.
    """
    cmd = "uv run python src/fakeartdetector/train.py"

    # Append flags only if provided (Typer/Hydra will handle the rest)
    if lr:
        cmd += f" --lr {lr}"
    if epochs:
        cmd += f" --epochs {epochs}"

    print(f"Executing: {cmd}")
    c.run(cmd)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate saved model data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py models/model.pth", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


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
@task
def visualize(ctx: Context) -> None:
    """Visualize the saved model preds."""
    ctx.run(f"uv run src/{PROJECT_NAME}/visualize.py models/model.pth", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def make_req_txt(ctx: Context) -> None:
    """Export dependencies to requirements.txt and requirements_dev.txt.

    - requirements.txt: runtime dependencies (default group)
    - requirements_dev.txt: runtime + dev dependencies
    """
    ctx.run("uv export --format requirements-txt --no-hashes -o requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run(
        "uv export --format requirements-txt --no-hashes --group dev -o requirements_dev.txt",
        echo=True,
        pty=not WINDOWS,
    )


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
