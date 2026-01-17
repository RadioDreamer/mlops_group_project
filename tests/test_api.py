import os
from io import BytesIO
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Import the app instance
from fakeartdetector.api import app


@pytest.fixture
def client():
    """Provides a TestClient and ensures GCS is disabled for local testing."""
    # Patch environment variables before the lifespan starts
    with patch.dict(os.environ, {"GCS_BUCKET_NAME": "", "MODEL_FILE": ""}):
        with TestClient(app) as c:
            yield c


def test_root(client):
    """Check health endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "OK"


def test_model_info(client):
    """Verify that the model source is local/random and not GCS."""
    response = client.get("/model-info")
    assert response.status_code == 200
    source = response.json()["loaded_model_source"]
    # Ensure we didn't use GCS
    assert "gs://" not in source
    assert "device" in response.json()


@patch("fakeartdetector.api.model")
def test_model_inference(mock_model, client):
    """
    Test the inference endpoint with a dummy image.
    Mocks the model call to avoid actual compute and ensure stability.
    """
    # 1. Setup Mock: Model returns a logit of 2.0 (Prob > 0.5)
    mock_model.return_value = torch.tensor([[2.0]])

    # 2. Create a dummy image in memory
    img = Image.new("RGB", (32, 32), color="red")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    # 3. Execution
    response = client.post("/model/", files={"data": ("test_image.jpg", img_byte_arr, "image/jpeg")})

    # 4. Assertions
    assert response.status_code == 200
    data = response.json()
    assert "isAI" in data
    assert "probability" in data
    assert data["isAI"] is True  # Because 2.0 logit -> >0.5 prob
