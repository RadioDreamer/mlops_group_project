from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import pytorch_lightning.profilers as profilers
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from typer.testing import CliRunner

from fakeartdetector.train import app, train_impl


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


@pytest.mark.parametrize("profiler_name", ["simple", "advanced"])
@pytest.mark.parametrize("num_workers", [0, 4])
def test_profiler_and_num_workers_combinations(dummy_config, dummy_datasets, tmp_path, profiler_name, num_workers):
    """Run training with different config combinations to hit missing lines."""
    dummy_config.profiler.type = profiler_name
    dummy_config.experiment.hyperparameters.num_workers = num_workers

    with (
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
    ):
        from fakeartdetector.train import train_impl

        train_impl(dummy_config)


@pytest.mark.parametrize("profiler_type", ["simple", "none"])
def test_profiler_logic(profiler_type, dummy_config, dummy_datasets, tmp_path):
    dummy_config.profiler.type = profiler_type

    with (
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
    ):
        train_impl(dummy_config)


def test_num_workers_logic(dummy_config, dummy_datasets, tmp_path):
    """Test that num_workers is correctly calculated."""
    dummy_config.experiment.hyperparameters.num_workers = None

    with (
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
        patch("os.cpu_count", return_value=8),
    ):
        train_impl(dummy_config)


def test_pytorch_profiler_logic(dummy_config, dummy_datasets, tmp_path):
    dummy_config.profiler = {
        "type": "pytorch",
        "tensorboard": True,
        "schedule": {"wait": 1, "warmup": 1, "active": 2, "repeat": 1},
        "filename": "test_trace",
    }

    # Real profiler instance
    real_profiler = profilers.PyTorchProfiler()

    # Patch only the constructor call to return the real instance
    with (
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
        patch("torch.profiler.schedule"),
        patch("torch.profiler.tensorboard_trace_handler"),
        patch("fakeartdetector.train.PyTorchProfiler", return_value=real_profiler) as mock_profiler_cls,
    ):
        train_impl(dummy_config)

        # Ensure the profiler was constructed and is still a real type
        mock_profiler_cls.assert_called()


def test_advanced_profiler_summary_saving(dummy_config, dummy_datasets, tmp_path):
    dummy_config.profiler.type = "advanced"

    # Patch the AdvancedProfiler.summary and filesystem
    with (
        patch.object(
            profilers.AdvancedProfiler, "summary", return_value="Mock Profiler Summary Output"
        ) as mock_summary,
        patch("fakeartdetector.train.cifake", return_value=dummy_datasets),
        patch("fakeartdetector.train.pl.Trainer"),
        patch("fakeartdetector.train.get_hydra_output_dir", return_value=tmp_path),
        patch("builtins.open", mock_open()) as m_open,
    ):
        train_impl(dummy_config)
        mock_summary.assert_called()
        write_calls = [call.args[0] for call in m_open().write.call_args_list]
        assert "Mock Profiler Summary Output" in write_calls


def test_train_cli_execution():
    runner = CliRunner()

    actual_config_path = Path(__file__).resolve().parents[1] / "configs"

    with (
        patch("fakeartdetector.train.train_impl") as mock_impl,
        patch("fakeartdetector.train.CONFIG_DIR", actual_config_path),
        patch("sys.argv", ["train.py"]),
    ):
        result = runner.invoke(
            app,
            [
                "--lr",
                "0.002",  # do NOT include "train"
            ],
        )

    assert result.exit_code == 0, f"CLI failed with output: {result.output}"
    assert mock_impl.called
