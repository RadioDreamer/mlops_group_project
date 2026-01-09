import sys
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from loguru import logger


def get_hydra_output_dir() -> Path:
	"""Return Hydra's per-run output directory.

	Raises a helpful error if Hydra hasn't been initialized.
	"""
	try:
		return Path(HydraConfig.get().runtime.output_dir)
	except Exception as exc:  # pragma: no cover
		raise RuntimeError(
			"Hydra is not initialized. Run via the Typer/Hydra CLI entrypoints (e.g. `python -m fakeartdetector.train`) "
			"so Hydra can compose config and set the runtime output dir."
		) from exc


def configure_loguru_file(
	output_dir: Path,
	*,
	filename: str,
	rotation: str,
	also_stderr: bool = False,
	stderr_level: str = "INFO",
) -> Path:
	"""Configure Loguru sinks.

	- Removes existing sinks to avoid duplicates.
	- Writes a rotating log file to `output_dir/filename`.
	- Optionally also logs to stderr.
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	log_path = output_dir / filename

	logger.remove()
	if also_stderr:
		logger.add(sys.stderr, level=stderr_level)
	logger.add(str(log_path), rotation=rotation)
	return log_path


def resolve_path(path_str: str, *, base_dir: Path | None = None) -> Path:
	"""Resolve a possibly relative path against `base_dir` (default: cwd)."""
	path = Path(path_str)
	if path.is_absolute():
		return path
	return (base_dir or Path.cwd()) / path

