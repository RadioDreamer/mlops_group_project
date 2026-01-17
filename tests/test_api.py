import os
from io import BytesIO
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from fakeartdetector.api import app


@pytest.fixture
def client():
    """Provides a TestClient while ensuring GCS is disabled for local testing."""
    # Force local mode by clearing GCS environment variables
    with patch.dict(os.environ, {"GCS_BUCKET_NAME": "", "MODEL_FILE": ""}):
        with TestClient(app) as c:
            yield c


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


@patch("fakeartdetector.api.model")
def test_model_inference(mock_model, client):
    """Test the inference endpoint with a dummy image."""
    # Mock model to return a specific logit
    mock_model.return_value = torch.tensor([[10.0]])

    # Create dummy image bytes
    img = Image.new("RGB", (32, 32), color="red")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")

    response = client.post("/model/", files={"data": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    assert "isAI" in response.json()
