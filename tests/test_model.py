import pytest
import torch

from fakeartdetector.model import FakeArtClassifier  # Adjust import based on your folder structure


@pytest.fixture
def model():
    """Fixture to create a fresh model for each test."""
    return FakeArtClassifier()


@pytest.fixture
def sample_batch():
    """Fixture to create a dummy batch (images and labels)."""
    batch_size = 4
    channels, height, width = 3, 32, 32

    # Random images: [Batch, 3, 32, 32]
    x = torch.randn(batch_size, channels, height, width)

    # Random binary labels: [Batch, 1] (Float type required for BCEWithLogitsLoss)
    y = torch.randint(0, 2, (batch_size, 1)).float()

    return x, y


def test_model_output_shape(model, sample_batch):
    """Check if the model outputs the correct shape [Batch_Size, 1]."""
    x, _ = sample_batch
    output = model(x)

    assert output.shape == (4, 1), f"Expected shape (4, 1), but got {output.shape}"


def test_training_step_runs(model, sample_batch):
    """Check if a single training step runs without crashing and returns a valid loss."""
    loss = model.training_step(sample_batch, batch_idx=0)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"
    assert not torch.isnan(loss), "Loss returned NaN!"


def test_output_is_logit(model, sample_batch):
    """Check if output acts like a logit (can go beyond 0..1 range)."""
    with torch.no_grad():
        # Force a large output
        model.head.weight.fill_(10.0)
        model.head.bias.fill_(100.0)

    output = model(sample_batch[0])

    # If this was Sigmoid, max would be 1.0.
    # Since it is a logit, it will be huge (approx 100+).
    assert output.max() > 1.0, "Output is constrained to [0,1], likely a Sigmoid activation exists!"


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_variable_batch_sizes(model, batch_size):
    """Ensure the model handles different batch sizes correctly."""
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    assert output.shape == (batch_size, 1)
