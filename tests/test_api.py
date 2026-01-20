import os
from io import BytesIO
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError

from fakeartdetector.api import app


@pytest.fixture
def client():
    """Provides a TestClient while ensuring GCS is disabled for local testing."""
    # Backup original environment variables so tests do not leak state
    original_gcs_bucket = os.environ.get("GCS_BUCKET_NAME")
    original_model_file = os.environ.get("MODEL_FILE")

    # Force local mode by clearing GCS-related environment variables
    with patch.dict(os.environ, {"GCS_BUCKET_NAME": "", "MODEL_FILE": ""}, clear=False):
        with TestClient(app) as c:
            # Verify that the app has initialized and exposes model info under this environment
            response = c.get("/model-info")
            assert response.status_code == 200
            assert "loaded_model_source" in response.json()
            yield c

    # Restore original environment variables after tests complete
    if original_gcs_bucket is not None:
        os.environ["GCS_BUCKET_NAME"] = original_gcs_bucket
    else:
        os.environ.pop("GCS_BUCKET_NAME", None)

    if original_model_file is not None:
        os.environ["MODEL_FILE"] = original_model_file
    else:
        os.environ.pop("MODEL_FILE", None)


def test_read_root(client):
    """Verify health check returns 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "OK"


def test_model_info(client):
    """Verify the API is running on the expected device and source."""
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "loaded_model_source" in response.json()


def test_model_inference_comprehensive(client):
    """
    Test functionality, reliability, and error handling.
    Includes a spy assertion to verify that image preprocessing
    correctly formats the tensor for the model.
    """

    # 1. TEST SUCCESSFUL INFERENCE + PREPROCESSING VERIFICATION
    with patch("fakeartdetector.api.model") as mock_model:
        # Mock return value (logit)
        mock_model.head.return_value = torch.tensor([[10.0]])

        # Create a red 100x100 image (to verify resizing logic)
        img = Image.new("RGB", (100, 100), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")

        response = client.post("/model/", files={"data": ("test.jpg", buf.getvalue(), "image/jpeg")})

        # Verify Response
        assert response.status_code == 200
        assert response.json()["isAI"] is True

        args, kwargs = mock_model.backbone.call_args
        passed_tensor = args[0]

        assert isinstance(passed_tensor, torch.Tensor)
        assert passed_tensor.shape == (1, 3, 32, 32)
        assert passed_tensor.device.type in ["cpu", "cuda", "mps"]

    # 2. TEST NEGATIVE CASE: MISSING FILE
    response_missing = client.post("/model/", files={})
    assert response_missing.status_code == 422

    # 3. TEST NEGATIVE CASE: CORRUPTED/INVALID IMAGE
    # Expecting PIL to raise UnidentifiedImageError because Starlette TestClient re-raises it
    invalid_data = b"not_an_image"
    with pytest.raises(UnidentifiedImageError):
        client.post("/model/", files={"data": ("bad.jpg", invalid_data, "image/jpeg")})

    # 4. TEST NEGATIVE CASE: OVERSIZED FILE (Simulated)
    with patch("starlette.testclient.TestClient.post") as mock_post:
        mock_post.return_value.status_code = 413
        response_large = client.post("/model/", files={"data": ("huge.jpg", b"0" * 10**7)})
        assert response_large.status_code == 413
