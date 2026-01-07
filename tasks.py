import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "fakeartdetector"
PYTHON_VERSION = "3.11"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


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


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


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
