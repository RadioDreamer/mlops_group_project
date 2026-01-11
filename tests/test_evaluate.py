from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from fakeartdetector.evaluate import evaluate_checkpoint, evaluate_impl


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


# --- ✅ FIXED VERSION ---
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


# --- ✅ FIXED VERSION ---
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
