# fakeartdetector

An MLOps project for classifying images as **Real** vs **AI-generated**.

This documentation covers:

- How to install and run the project locally
- The main workflows (data preprocessing, training, docs)
- Auto-generated API reference for the `fakeartdetector` Python package

## Quick links

- Getting started: see [Getting Started](getting-started.md)
- Day-to-day commands: see [Workflows](workflows.md)
- Code reference: see the **API Reference** section in the navigation

## What’s in the repo

- `src/fakeartdetector/`: Python package
- `data/`: DVC-managed data (processed tensors live in `data/processed/`)
- `dockerfiles/`: Dockerfiles for training and API images
- `tasks.py`: Invoke tasks (recommended entrypoint for common commands)
