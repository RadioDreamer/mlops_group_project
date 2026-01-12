import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Dataset

from fakeartdetector.data import cifake, normalize, preprocess_data
from tests import _PATH_DATA


@pytest.fixture
def dummy_images():
    # 4 images, 3 channels, 32x32
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def dummy_labels():
    return torch.tensor([1, 0, 1, 0])


def test_normalize(dummy_images):
    normed = normalize(dummy_images)
    # Check shape unchanged
    assert normed.shape == dummy_images.shape
    # Check mean ~0 and std ~1 per channel
    mean = normed.mean(dim=(0, 2, 3))
    std = normed.std(dim=(0, 2, 3))
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-6)


def test_cifake_raises_error_on_missing_files():
    """Ensure FileNotFoundError is raised if the processed data doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Preprocessed data file"):
        cifake(processed_dir="non_existent_directory")


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
    # Check that both train and test sets contain only binary labels 0 and 1
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


def test_cifake_full_execution_flow(tmp_path):
    """
    Test the complete loading logic by creating real temporary .pt files.
    This ensures the full CIFAKE data loading and tensor dataset creation
    logic in data.py is exercised.
    """
    # 1. Setup: Create the directory structure in the pytest temp folder
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    # 2. Create dummy tensors and save them to the expected paths
    dummy_tensor = torch.randn(2, 3, 32, 32)
    dummy_label = torch.tensor([1, 0])

    torch.save(dummy_tensor, processed_dir / "train_images.pt")
    torch.save(dummy_label, processed_dir / "train_target.pt")
    torch.save(dummy_tensor, processed_dir / "test_images.pt")
    torch.save(dummy_label, processed_dir / "test_target.pt")

    # 3. Execution: Call cifake pointing to our temp directory
    train_set, test_set = cifake(processed_dir=str(processed_dir))

    # 4. Assertions: Verify data integrity
    assert len(train_set) == 2
    assert len(test_set) == 2
    assert isinstance(train_set, torch.utils.data.TensorDataset)
    assert train_set[0][0].shape == (3, 32, 32)


@patch("torch.save")
@patch("typer.echo")
@patch("datasets.load_dataset")
def test_preprocess_data(mock_load_dataset, mock_echo, mock_save, tmp_path):
    # Mock dataset to return 2 items for train and test
    mock_dataset = {
        "train": [
            {"image": MagicMock(convert=lambda x: MagicMock()), "label": 1},
            {"image": MagicMock(convert=lambda x: MagicMock()), "label": 0},
        ],
        "test": [{"image": MagicMock(convert=lambda x: MagicMock()), "label": 1}],
    }
    mock_load_dataset.return_value = mock_dataset

    # Patch torchvision transforms to return dummy tensor
    with patch("torchvision.transforms.Compose", lambda x: lambda img: torch.randn(3, 32, 32)):
        preprocess_data(str(tmp_path))

    # Should call save 4 times: train_images, train_target, test_images, test_target
    assert mock_save.call_count == 4
    # Echo should be called at least 3 times (start, train, test, finish)
    assert mock_echo.call_count >= 3


@patch("torch.load")
def test_cifake_success(mock_load, tmp_path):
    # Create dummy files
    (tmp_path / "train_images.pt").touch()
    (tmp_path / "train_target.pt").touch()
    (tmp_path / "test_images.pt").touch()
    (tmp_path / "test_target.pt").touch()

    # Mock torch.load to return tensors
    mock_load.side_effect = [
        torch.randn(4, 3, 32, 32),  # train images
        torch.tensor([1, 0, 1, 0]),  # train target
        torch.randn(2, 3, 32, 32),  # test images
        torch.tensor([1, 0]),  # test target
    ]

    train_set, test_set = cifake(str(tmp_path))

    # Check dataset types
    assert isinstance(train_set, torch.utils.data.TensorDataset)
    assert isinstance(test_set, torch.utils.data.TensorDataset)
    # Check lengths
    assert len(train_set) == 4
    assert len(test_set) == 2


def test_cifake_missing_file(tmp_path):
    # No files created -> should raise
    with pytest.raises(FileNotFoundError):
        cifake(str(tmp_path))


def test_show_image_and_target(dummy_images, dummy_labels):
    # Patch plt.show to prevent GUI and allow assertion
    with patch("matplotlib.pyplot.show") as mock_show:
        from fakeartdetector.data import show_image_and_target

        show_image_and_target(dummy_images, dummy_labels)

        # Ensure plt.show() was called once
        mock_show.assert_called_once()
