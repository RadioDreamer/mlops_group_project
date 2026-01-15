import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BytesIO
from typing import cast

import torchvision.transforms as T  # noqa:N812
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
from PIL import Image
from torch import Tensor, cuda, device, load, no_grad, sigmoid, unsqueeze
from torch.backends import mps

from fakeartdetector.model import FakeArtClassifier


def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model and gets ready for inference"""
    global model, DEVICE, transform
    print("Hello, i am loading the model")
    DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")
    model = FakeArtClassifier().to(DEVICE)

    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    model_file = os.environ.get("MODEL_FILE")
    local_model_path = "model.pth"

    state_dict = None

    if bucket_name and model_file:
        try:
            print(f"Attempting to download model from gs://{bucket_name}/{model_file}")
            download_model(bucket_name, model_file, local_model_path)
            state_dict = load(local_model_path, map_location=DEVICE)
            print("Successfully loaded downloaded model")
        except Exception as e:
            print(f"Failed to download or load model from GCS: {e}")
            print("Using fallback model provided in image")
    
    if state_dict is None:
         # Fallback to local base model
         if os.path.exists("./models/base_model.pth"):
            print("Loading base_model.pth from image")
            state_dict = load("./models/base_model.pth", map_location=DEVICE)
         else:
            print("No model found! Initializing random weights.")

    if state_dict:
        model.load_state_dict(state_dict)

    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    )

    print(f"Model: {model}\n")
    if state_dict:
        for i, key in enumerate(state_dict.keys()):
            print(f"{i}: {key}")
    print("Model state dict loaded successfully\n")

    yield
    print("Cleaning up here.....")
    del model, DEVICE
    print("Goodbye")


app = FastAPI(lifespan=lifespan)


# ------------------------------------------ GET ------------------------------------------
@app.get("/")
def root():
    """Health Check."""
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}


@app.get("/models")
def int_item():
    """http://127.0.0.1:8000/models"""
    model_list = os.listdir("./models")
    return model_list


@app.get("/model-info")
def model_info():
    """Returns model and device information"""
    return {
        "device": str(DEVICE),
        "model": str(model),
        "model_device": str(next(model.parameters()).device),
    }


# ------------------------------------------ POST ------------------------------------------
@app.post("/model/")
async def model_inference(data: UploadFile = File(...)):
    """Answers if the Image Sent is AI art of not

    Example Usage:
        curl -X POST "http://localhost:8000/model/" \
            -F "data=@cat.jpg" \
        """

    image_bytes = await data.read()
    img = await image_clean_utility(image_bytes)
    with no_grad():
        logits = model(img)
        prob = sigmoid(logits)
        is_ai = (prob > 0.5).item()
    return {"isAI": is_ai, "probability": prob.item()}


async def image_clean_utility(image_bytes: bytes):
    i_image = Image.open(BytesIO(image_bytes))
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    image = cast(Tensor, transform(i_image))
    return cast(Tensor, unsqueeze(image, 0).float().to(DEVICE))
