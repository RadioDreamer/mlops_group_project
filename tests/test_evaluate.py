from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
from typer.testing import CliRunner

from fakeartdetector.data import cifake
from fakeartdetector.evaluate import app, evaluate_checkpoint, evaluate_impl


@patch("torch.load")
@patch("pathlib.Path.is_file")
def test_data_loading_logic(mock_is_file, mock_torch_load):
    """Force execution of cifake() by mocking file existence and torch loading."""

    # Make all file checks succeed
    mock_is_file.return_value = True

    # Make torch.load return dummy tensors
    dummy_tensor = torch.randn(1, 3, 32, 32)
    dummy_target = torch.tensor([1])
    mock_torch_load.side_effect = [dummy_tensor, dummy_target, dummy_tensor, dummy_target]

    # Call the function
    train_set, test_set = cifake()

    # Assertions
    assert isinstance(train_set, torch.utils.data.TensorDataset)
    assert isinstance(test_set, torch.utils.data.TensorDataset)
    assert mock_torch_load.call_count == 4


@pytest.fixture
def dummy_test_set():
    """Create a minimal 4-image dataset."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([1, 0, 1, 0])
    return TensorDataset(images, labels)


@pytest.fixture
def dummy_cfg(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.touch()
    return OmegaConf.create({"evaluate": {"model_checkpoint": str(ckpt), "batch_size": 4, "threshold": 0.5}})


@patch("fakeartdetector.evaluate.cifake")
@patch("fakeartdetector.evaluate.load")
@patch("fakeartdetector.evaluate.FakeArtClassifier")
def test_evaluate_checkpoint_logic(mock_model_class, mock_load, mock_cifake, dummy_test_set):
    mock_cifake.return_value = (None, dummy_test_set)
    mock_load.return_value = {}

    mock_logits = torch.tensor([[10.0], [-10.0], [10.0], [-10.0]])

    # Define a simple callable fake model
    class DummyModel:
        def __call__(self, x):
            return mock_logits

        def to(self, device):
            return self

        def eval(self):
            return None

        def load_state_dict(self, state_dict):
            return None

    mock_model_class.return_value = DummyModel()

    accuracy = evaluate_checkpoint("dummy_ckpt.pt", batch_size=4, threshold=0.5)

    assert accuracy == 1.0


def test_threshold_logic_direct(dummy_test_set):
    logits = torch.tensor([[0.1], [0.1], [0.1], [0.1]])

    class DummyModel:
        def __call__(self, x):
            return logits

        def to(self, device):
            return self

        def eval(self):
            return None

        def load_state_dict(self, state_dict):
            return None

    with (
        patch("fakeartdetector.evaluate.FakeArtClassifier", return_value=DummyModel()),
        patch("fakeartdetector.evaluate.cifake", return_value=(None, dummy_test_set)),
        patch("fakeartdetector.evaluate.load"),
    ):
        acc_low = evaluate_checkpoint("fake.pt", batch_size=4, threshold=0.5)
        assert acc_low == 0.5

        acc_high = evaluate_checkpoint("fake.pt", batch_size=4, threshold=0.6)
        assert acc_high == 0.5


@patch("fakeartdetector.evaluate.evaluate_checkpoint")
@patch("fakeartdetector.evaluate.get_hydra_output_dir")
@patch("fakeartdetector.evaluate.configure_loguru_file")
@patch("fakeartdetector.evaluate.resolve_path")
def test_evaluate_impl_execution(mock_resolve, mock_log, mock_dir, mock_eval_func, dummy_cfg, tmp_path):
    mock_dir.return_value = tmp_path
    mock_resolve.return_value = Path(dummy_cfg.evaluate.model_checkpoint)
    mock_eval_func.return_value = 0.99

    evaluate_impl(dummy_cfg)
    mock_eval_func.assert_called_once()


def test_evaluate_cli_overrides(tmp_path):
    """Ensure CLI runs without Hydra interference."""
    runner = CliRunner()

    # Outputs configs!!
    # Dummy Hydra config dir (Hydra otherwise errors)
    dummy_config_dir = tmp_path / "configs"
    dummy_config_dir.mkdir(parents=True)

    # Mock everything that could touch the filesystem or Hydra
    with (
        patch("fakeartdetector.evaluate.CONFIG_DIR", dummy_config_dir),
        patch("fakeartdetector.evaluate.evaluate_impl") as mock_impl,
        patch("fakeartdetector.evaluate.resolve_path", side_effect=lambda x: Path(x)),
        patch("fakeartdetector.evaluate.get_hydra_output_dir", return_value=tmp_path),
        patch("fakeartdetector.evaluate.configure_loguru_file", return_value="dummy.log"),
        patch("fakeartdetector.evaluate.hydra.main") as mock_hydra,
        patch("sys.argv", ["evaluate.py"]),
    ):
        # Fake the hydra decorator to just call the wrapped function
        def fake_hydra_main(*args, **kwargs):
            def wrapper(fn):
                def wrapped():
                    cfg = {"evaluate": {"model_checkpoint": "dummy.pt", "batch_size": 16, "threshold": 0.7}}
                    mock_impl(cfg)

                return wrapped

            return wrapper

        mock_hydra.side_effect = fake_hydra_main
        result = runner.invoke(
            app,
            [
                "--batch-size",
                "16",
                "--threshold",
                "0.7",
            ],
        )

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0, f"CLI failed with: {result.output}"
    mock_impl.assert_called_once()
