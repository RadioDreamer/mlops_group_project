# AI Art Detector

## Goal of the project
The rapid development of AI generated images has created an environment where it has become incredibly difficult to tell apart real images from synthesized ones. The goal of our project is to create a robust, End-to-End pipeline for determining whether images are 'Real' or 'AI generated'. We have decided to use this project as the topic is relevant and the output of the model is easily interpretable for images. Finally, it allows us to play around with most of the concepts and technologies outlined in the course.

## Frameworks and Third-Party Integrations
The core deep library used will be PyTorch. To extend its capabilities, we will be using the Hugging Face ecosystem's `datasets` library. Additionally, the pretrained neural network architectures will be imported from `timm` library. This will allow us to not spend time on defining and finetuning models and focus on improving the pipeline and operationalizing the task.

## Data
We will rely on the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset a collection of 60k real images (from CIFAR-10) and 60k AI-generated images (created via Stable Diffusion 1.4). There are 100k images for training (50k per class) and 20k for testing (10k per class). We will be using the 32x32 version of this dataset. This choice will enable us to rapidly iterate on our pipeline and minimizes storage required on GCP. However, it is still powerful enough to test for potential drifting in the distribution. We will manage this dataset using DVC backed by a Google Cloud Storage bucket.

## Models
Our initial model will be using a standard CNN model. Additionally, once we feel comfortable that the pipeline is efficiently set up, we will explore pretrained models such as ResNet18 and also transformer based models (such as ViT). Once we found the best model, we will package it into a Docker container and deploy it with an inference API using FastAPI.

## References
1.  **CIFAR-10 Dataset:**
    * Krizhevsky, A., & Hinton, G. (2009). *Learning multiple layers of features from tiny images*.
    * [Source](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

2.  **CIFAKE Dataset:**
    * Bird, J.J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.
    * [Source](https://ieeexplore.ieee.org/document/10403361)

## Project structure

The directory structure of the project looks like this:
```txt
.
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
│   ├── processed
│   └── raw
├── configs/                  # Configuration files
│   ├── dataset
│   │   └── base.yaml
│   ├── default_config.yaml
│   ├── evaluate
│   │   └── base.yaml
│   ├── experiment
│   │   ├── base.yaml
│   │   └── trials1.yaml
│   ├── logging
│   │   └── base.yaml
│   └── optimizer
│       ├── adam.yaml
│       └── sgd.yaml
├── data/                     # Data directory
│   ├── logs
│   │   └── train_logger.log
│   ├── processed
│   │   ├── test_images.pt
│   │   ├── test_target.pt
│   │   ├── train_images.pt
│   │   └── train_target.pt
│   └── raw
├── data.dvc
├── dockerfiles               # Docker Files
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # Documentation
│   ├── build/
│   ├── mkdocs.yaml
│   ├── README.md
│   └── source/
│       ├── getting-started.md
│       ├── index.md
│       ├── reference
│       │   ├── api.md
│       │   ├── data.md
│       │   ├── evaluate.md
│       │   ├── model.md
│       │   ├── train.md
│       │   └── visualize.md
│       └── workflows.md
├── experiments
│   └── config.yaml
├── LICENSE
├── models/                   # Trained models
│   └── base_model.pth
├── notebooks
├── outputs
│   └── 2026-01-09
│       ├── 18-35-44
│       │   ├── train_hydra.log
│       │   └── train.log
│       ├── 18-37-54/
│       ├                     # rest of the logs from hydra
│
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
├── reports/                  # Reports
│   └── figures
├── src/                      # Source code
│   ├── fakeartdetector
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── helpers.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
│   └── fakeartdetector.egg-info
│       ├── dependency_links.txt
│       ├── entry_points.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
├── tests                     # Tests
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── tasks.py                  # Project tasks
└── uv.lock

50 directories, 129 files
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
