# AI Art Detector 

## Goal of the project
The rapid development of AI generated images has created an environment where it has become incredibly difficult to tell apart real images from synthesized ones. The goal of our project is to create a robust, End-to-End pipeline for determining whether images are 'Real' or 'AI generated'.

## Frameworks and Third-Party Integrations
The core deep library used will be PyTorch. To extend its capabilities, we will be using the Hugging Face ecosystem's `datasets` library. Additionally, the pretrained neural network architectures will be imported from `timm` library. This will allow us to not spend time on defining and finetuning models and focus on operationalizing the pipeline.

## Data
We will rely on the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset a collection of 60k real images (from CIFAR-10) and 60k AI-generated images (created via Stable Diffusion 1.4). There are 100k images for training (50k per class) and 20k for testing (10k per class). We will be using the 32x32 version of this dataset. This choice will enable us to rapidly iterate on our pipeline and minimizes storage required on GCP. However, it is still powerful enough to test for potential drifting in the distribution.

## Models
Our initial model will be using a pretrained ResNet18 model. Additionally, once we feel comfortable that the pipeline is efficiently set up, we will explore Transformer based models (such as ViT).

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
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
