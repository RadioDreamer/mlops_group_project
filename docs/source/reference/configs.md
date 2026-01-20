# `Configs`

This project uses a `configs/` directory to hold YAML configuration files used by training, evaluation, and experiments.

## Overview

- `configs/default_config.yaml`: base configuration values.
- `configs/dataset/`: dataset-specific configuration (e.g., `base.yaml`, `cloud.yaml`).
- `configs/experiment/`: experiment specifications and multi-trial definitions.
- `configs/evaluate/`, `configs/logging/`, `configs/optimizer/`, `configs/profiler/`: modular configs for their respective subsystems.

## How to use

- The training/evaluation scripts pick up the correct config via Hydra or direct parsing. See `src/fakeartdetector/train.py` and `src/fakeartdetector/evaluate.py` for how configs are loaded.
- To run an experiment locally, edit or override values with CLI flags as supported by the scripts.

## Best practices

- Keep sensitive values (API keys, GCS bucket names) in environment variables or a `.env` file and avoid committing them to the repo.
- Use the `configs/experiment/trials` files to run sweep experiments.

## Example (override)

```bash
python src/fakeartdetector/train.py --config configs/default_config.yaml
```
