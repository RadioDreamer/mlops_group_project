from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from fakeartdetector.train import train_impl


@pytest.fixture
def dummy_config(tmp_path):
    """Creates a minimal Hydra-style configuration for testing."""
    save_path = tmp_path / "model.pt"
    processed_dir = tmp_path / "data"
    processed_dir.mkdir()

    config_dict = {
        "experiment": {
            "hyperparameters": {
                "lr": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "seed": 42,
                "dropout": 0.1,
                "num_workers": 0,
                "precision": "32",
            }
        },
        "dataset": {"dataset": {"path": str(processed_dir)}, "savedTo": {"path": str(save_path)}},
        "optimizer": None,  # Falls back to default Adam
        "logging": {"base": {}},
        "profiler": {"type": "none"},
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def dummy_datasets():
    """Creates small dummy datasets to avoid loading CIFAKE from disk."""
    img = torch.randn(4, 3, 32, 32)
    label = torch.randint(0, 2, (4,))
    train_set = TensorDataset(img, label)
    val_set = TensorDataset(img, label)
    return train_set, val_set


@patch("fakeartdetector.train.cifake")
@patch("fakeartdetector.train.pl.Trainer")
@patch("fakeartdetector.train.get_hydra_output_dir")
@patch("fakeartdetector.train.configure_loguru_file")
def test_train_impl_smoke(
    mock_loguru, mock_hydra_dir, mock_trainer_class, mock_cifake, dummy_config, dummy_datasets, tmp_path
):
    """
    Smoke test for train_impl.
    It mocks external dependencies to ensure the logic flows without errors.
    """
    # Setup mocks
    mock_hydra_dir.return_value = tmp_path
    mock_loguru.return_value = tmp_path / "train.log"
    mock_cifake.return_value = dummy_datasets

    # Mock the trainer instance
    mock_trainer_instance = MagicMock()
    mock_trainer_class.return_value = mock_trainer_instance

    # Run the implementation
    train_impl(dummy_config)

    # Assertions: Did it call the essential parts?
    mock_cifake.assert_called_once()
    mock_trainer_instance.fit.assert_called_once()

    # Check if artifacts were saved
    assert (tmp_path / "artifacts.yaml").exists(), "Artifacts manifest was not saved"
    assert (tmp_path / "config_full.yaml").exists(), "Full config was not saved"


@pytest.mark.parametrize("profiler_type", ["simple", "none"])
def test_profiler_logic(profiler_type, dummy_config, dummy_datasets, tmp_path):
    dummy_config.profiler.type = profiler_type

    # USE dummy_datasets instead of (MagicMock(), MagicMock())
    with (
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
    ):
        train_impl(dummy_config)
        # If no error is raised, the branching logic for the profiler is valid.


def test_num_workers_logic(dummy_config, dummy_datasets, tmp_path):
    """Test that num_workers is correctly calculated."""
    dummy_config.experiment.hyperparameters.num_workers = None

    # Pass dummy_datasets (which has 4 samples) instead of MagicMocks
    with (
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
        patch("os.cpu_count", return_value=8),
    ):
        train_impl(dummy_config)
        # No error means DataLoader accepted the dataset length
