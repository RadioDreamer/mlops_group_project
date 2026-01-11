import os

import pytest
import torch
from torch.utils.data import Dataset

from fakeartdetector.data import cifake, normalize
from tests import _PATH_DATA


def test_normalize():
    """Test that normalization correctly scales dummy image tensors."""
    # Create a dummy batch: [batch, channels, height, width]
    dummy_input = torch.randn(4, 3, 32, 32) + 5  # Shifted mean
    normalized = normalize(dummy_input)

    # Check that output has the same shape
    assert normalized.shape == dummy_input.shape, "Normalization changed the tensor shape"

    # Check that mean is close to 0 and std is close to 1 per channel
    for c in range(3):
        assert torch.abs(normalized[:, c, :, :].mean()) < 1e-5, f"Mean for channel {c} is not 0"
        assert torch.abs(normalized[:, c, :, :].std() - 1.0) < 1e-5, f"Std for channel {c} is not 1"


def test_cifake_raises_error_on_missing_files():
    """Ensure FileNotFoundError is raised if the processed data doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Preprocessed data file"):
        cifake(processed_dir="non_existent_directory")


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found in data/processed")
def test_data_loading():
    """Tests the integrity of the loaded CIFAKE dataset."""
    train_set, test_set = cifake(_PATH_DATA)

    # 1. Check types
    assert isinstance(train_set, Dataset), "Train set is not a torch Dataset"
    assert isinstance(test_set, Dataset), "Test set is not a torch Dataset"

    # 2. Check shapes of a sample
    img, label = train_set[0]
    assert img.shape == (3, 32, 32), "Image shape is not [3, 32, 32]"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"

    # 3. Check labels (should be 0 or 1 for Real vs Fake)
    # Using parametrization logic for specific label checks
    for dataset_name, dataset in [("train", train_set), ("test", test_set)]:
        # Extract all labels from the dataset tensors for a quick check
        labels = dataset.tensors[1]
        unique_labels = torch.unique(labels)
        assert torch.equal(unique_labels.sort()[0], torch.tensor([0, 1])), (
            f"Labels in {dataset_name} set are not exactly {0, 1}"
        )


@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_normalize_batch_invariance(batch_size):
    """Ensure normalization logic holds regardless of batch size."""
    x = torch.randn(batch_size, 3, 32, 32)
    y = normalize(x)
    assert y.shape[0] == batch_size, f"Batch size mismatch for N={batch_size}"
