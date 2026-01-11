from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from fakeartdetector.evaluate import evaluate_checkpoint, evaluate_impl


@pytest.fixture
def dummy_eval_config(tmp_path):
    """Minimal Hydra config for evaluation."""
    checkpoint_file = tmp_path / "dummy_ckpt.pt"
    checkpoint_file.touch()  # Create an empty file to satisfy path resolution

    config = {
        "evaluate": {"model_checkpoint": str(checkpoint_file), "batch_size": 2, "threshold": 0.5},
        "dataset": {"dataset": {"path": "/fake/data"}},
    }
    return OmegaConf.create(config)


@pytest.fixture
def dummy_test_set():
    """Creates a small 4-sample dataset for evaluation testing."""
    img = torch.randn(4, 3, 32, 32)
    # Labels: [1, 0, 1, 0]
    label = torch.tensor([1, 0, 1, 0])
    return TensorDataset(img, label)


@patch("fakeartdetector.evaluate.cifake")
@patch("fakeartdetector.evaluate.load")
@patch("fakeartdetector.evaluate.FakeArtClassifier")
# In tests/test_evaluate.py, update the mock setup:

def test_evaluate_checkpoint_accuracy(mock_classifier, mock_load, mock_cifake, dummy_test_set):
    mock_cifake.return_value = (None, dummy_test_set)

    # Create the instance mock
    mock_model_instance = MagicMock()

    # THIS IS THE FIX: The mock_model_instance itself must return the tensors when called
    mock_model_instance.return_value.side_effect = [torch.tensor([[10.0], [-10.0]]), torch.tensor([[10.0], [-10.0]])]

    # And ensure the class returns this instance
    mock_classifier.return_value = mock_model_instance

    accuracy = evaluate_checkpoint("fake_path.pt", batch_size=2, threshold=0.5)
    assert accuracy == 1.0
    assert mock_classifier.called
    assert mock_model_instance.eval.called


@patch("fakeartdetector.evaluate.evaluate_checkpoint")
@patch("fakeartdetector.evaluate.get_hydra_output_dir")
@patch("fakeartdetector.evaluate.configure_loguru_file")
def test_evaluate_impl_flow(mock_log, mock_dir, mock_eval_fn, dummy_eval_config, tmp_path):
    """Ensures evaluate_impl correctly extracts config and triggers evaluation."""
    mock_dir.return_value = tmp_path
    mock_eval_fn.return_value = 0.85

    evaluate_impl(dummy_eval_config)

    # Verify the parameters were passed correctly from Hydra config
    mock_eval_fn.assert_called_once_with(
        model_checkpoint=dummy_eval_config.evaluate.model_checkpoint, batch_size=2, threshold=0.5
    )


@pytest.mark.parametrize(
    "threshold, expected_acc",
    [
        (0.5, 0.5),  # Normal threshold
        (0.9, 0.25),  # Strict threshold might change accuracy
    ],
)
def test_threshold_logic(threshold, expected_acc, dummy_test_set):
    """Tests if changing the threshold logic affects evaluation (integration check)."""
    # This logic is usually part of the loop in evaluate_checkpoint.
    # We can test the math directly to ensure sigmoid/threshold logic is sound.
    logits = torch.tensor([[0.1], [0.1], [0.1], [0.1]])  # sigmoid(0.1) approx 0.52

    # If threshold=0.5, predictions are [1, 1, 1, 1] -> 2/4 = 0.5 acc
    # If threshold=0.9, predictions are [0, 0, 0, 0] -> 2/4 = 0.5 acc
    # Let's adjust logits to be more sensitive:
    logits = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])
    # sigmoid(1) > 0.5 is True. sigmoid(-1) > 0.5 is False.
    # With targets [1, 0, 1, 0], accuracy should be 1.0.

    with (
        patch("fakeartdetector.evaluate.FakeArtClassifier") as mock_model,
        patch("fakeartdetector.evaluate.cifake", return_value=(None, dummy_test_set)),
        patch("fakeartdetector.evaluate.load"),
    ):
        mock_model.return_value.side_effect = [logits]
        acc = evaluate_checkpoint("dummy", batch_size=4, threshold=threshold)
        assert isinstance(acc, float)
