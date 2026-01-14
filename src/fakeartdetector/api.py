import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BytesIO
from typing import cast

import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torch import Tensor, cuda, device, load, no_grad, sigmoid, unsqueeze
from torch.backends import mps

from fakeartdetector.model import FakeArtClassifier

database = {"username": [], "password": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model and gets ready for inference"""
    global model, DEVICE, transform
    print("Hello, i am loading the model")
    DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")
    model = FakeArtClassifier().to(DEVICE)
    state_dict = load("./models/base_model.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    )

    print(f"Model: {model}\n")
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
    # with open("image.jpg", "wb") as image:
    #    content = await data.read()
    #    image.write(content)
    # this would be a file response
    # return FileResponse(
    #    "resized_img.jpg",
    #    media_type="image/jpeg",
    #    filename="resized_img.jpg",
    #    status_code=HTTPStatus.OK,
    # )


async def image_clean_utility(image_bytes: bytes):
    i_image = Image.open(BytesIO(image_bytes))
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    image = cast(Tensor, transform(i_image))
    return cast(Tensor, unsqueeze(image, 0).float().to(DEVICE))
