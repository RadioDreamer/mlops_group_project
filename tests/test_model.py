import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from fakeartdetector.model import FakeArtClassifier

warnings.filterwarnings(
    "ignore",
    message=".*barebones=True.*",
    category=UserWarning,
)


@pytest.fixture
def model():
    """Fixture to create a fresh model for each test."""
    return FakeArtClassifier(lr=0.01, dropout=0.2)


@pytest.fixture
def sample_batch():
    """Fixture to create a dummy batch (images and labels)."""
    batch_size = 4
    channels, height, width = 3, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    # Binary labels for BCEWithLogitsLoss
    y = torch.randint(0, 2, (batch_size,)).float()
    return x, y


def test_model_output_shape(model, sample_batch):
    """Check if the model outputs the correct shape [Batch_Size, 1]."""
    x, _ = sample_batch
    output = model(x)
    assert output.shape == (4, 1), f"Expected shape (4, 1), but got {output.shape}"


def test_training_step_runs(model, sample_batch):
    """Check if a single training step runs without crashing and returns a scalar loss."""
    model.trainer = Mock()
    loss = model.training_step(sample_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss returned NaN"


def test_validation_step_runs(model, sample_batch):
    """Ensure validation logic is covered and produces a loss."""
    model.trainer = Mock()
    loss = model.validation_step(sample_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_output_is_logit(model, sample_batch):
    """Ensure no Sigmoid activation is applied to the final output."""
    with torch.no_grad():
        model.head.weight.fill_(10.0)
        model.head.bias.fill_(100.0)
    output = model(sample_batch[0])
    assert output.max() > 1.0, "Output is constrained to [0,1], suggesting a Sigmoid exists!"


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_variable_batch_sizes(model, batch_size):
    """Ensure the model handles various batch sizes."""
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 1)


def test_save_hyperparameters(model):
    """Verify that Lightning saves the correct hyperparameters."""
    assert model.hparams.lr == 0.01
    assert model.hparams.dropout == 0.2


def test_configure_optimizers_default(model):
    """Test the default optimizer path (Adam)."""
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.01


def test_configure_optimizers_with_config(model):
    """Test the Hydra-based optimizer path using a mock."""
    mock_cfg = MagicMock()
    model.optimizer_cfg = mock_cfg

    with patch("hydra.utils.instantiate") as mock_instantiate:
        model.configure_optimizers()
        mock_instantiate.assert_called_once()


def test_error_on_wrong_shape(model):
    """
    Assert that the model raises errors for invalid input shapes.
    Note: To make this pass, you should add validation logic to your forward() method.
    """
    # Test wrong number of dimensions (3D instead of 4D)
    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        model(torch.randn(3, 32, 32))

    # Test wrong number of channels
    with pytest.raises(ValueError, match="Expected 3 input channels"):
        model(torch.randn(1, 1, 32, 32))
