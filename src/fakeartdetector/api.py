import os
import shutil
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import cast

import torchvision.transforms as T  # noqa:N812
import wandb
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile
from google.cloud import storage
from PIL import Image
from torch import Tensor, cuda, device, load, no_grad, sigmoid, unsqueeze
from torch.backends import mps

from fakeartdetector.model import FakeArtClassifier
from fakeartdetector.sqlite_db import init_db, insert_row


def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")


def load_model_wandb(artifact_path):
    print(f"--- Loading Model from: {artifact_path} ---")
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

    # Access the specific artifact from the registry
    artifact = api.artifact(artifact_path)

    download_dir = "staged_model_dir"
    download_path = artifact.download(root=download_dir)

    # Get the checkpoint filename and load the model
    file_name = artifact.files()[0].name
    full_ckpt_path = os.path.join(download_path, file_name)

    # Cleanup: keep only the checkpoint we are about to load.
    try:
        staged_dir = Path(download_dir).resolve()
        keep_file = Path(full_ckpt_path).resolve()

        # Safety: only delete things inside staged_model_dir.
        if staged_dir in keep_file.parents and staged_dir.exists():
            for child in staged_dir.iterdir():
                # Keep the file itself, and the top-level directory that contains it (if any).
                if child == keep_file:
                    continue
                if child.is_dir() and keep_file in child.rglob(keep_file.name):
                    # Remove everything in this dir except the keep_file.
                    for sub in child.iterdir():
                        if sub == keep_file or (sub.is_dir() and keep_file in sub.rglob(keep_file.name)):
                            continue
                        if sub.is_dir():
                            shutil.rmtree(sub, ignore_errors=True)
                        else:
                            sub.unlink(missing_ok=True)
                    continue

                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: failed to cleanup staged_model_dir: {e}")

    return FakeArtClassifier.load_from_checkpoint(full_ckpt_path)


def add_to_database(
    latency: float,
    embedding: float,
    prediction: float,
) -> None:
    """Save input image and prediction to database."""
    try:
        row_id = insert_row(latency, embedding, prediction)
        print(f"Inserted row into sqlite db id={row_id}")
    except Exception as e:
        print(f"Error inserting row into sqlite db: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model and gets ready for inference"""
    # Ensure directories exist
    load_dotenv()
    global model, DEVICE, transform, loaded_model_source, api
    print("Hello, i am loading the model")
    DEVICE = device("cuda" if cuda.is_available() else "mps" if mps.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    # Initialize sqlite database for inference logs
    db_path = os.getenv("SQLITE_DB_PATH", "data/inference_logs/inference_logs.db")
    try:
        init_db(db_path)
        print(f"Initialized sqlite DB at: {db_path}")
    except Exception as e:
        print(f"Failed to initialize sqlite DB at {db_path}: {e}")

    model = FakeArtClassifier().to(DEVICE)
    loaded_model_source = "None"

    # Priority order:
    # 1. USE_LOCAL flag (local model path)
    # 2. MODEL_NAME env variable (wandb model)
    # 3. LOAD_FROM_BUCKET flag (GCS model)
    # 4. Fallback to local base_model.pth

    if os.getenv("USE_LOCAL"):
        print("Loading model from local path...")
        local_model_path = "models/base_model.pth"
        if os.path.exists(local_model_path):
            state_dict = load(local_model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            loaded_model_source = "Local base_model.pth"
            print("Successfully loaded local model")
        else:
            print(f"Local model not found at {local_model_path}")

    elif os.getenv("MODEL_NAME"):
        print("Loading model from wandb MODEL_NAME...")
        model = load_model_wandb(os.getenv("MODEL_NAME")).to(DEVICE)
        loaded_model_source = os.getenv("MODEL_NAME")

    elif os.getenv("LOAD_FROM_BUCKET") in ("true", "1", "yes"):
        bucket_name = os.environ.get("GCS_BUCKET_NAME")
        model_file = os.environ.get("MODEL_FILE")
        local_model_path = "model.pth"

        if bucket_name and model_file:
            try:
                print(f"Loading model from GCS: gs://{bucket_name}/{model_file}")
                download_model(bucket_name, model_file, local_model_path)
                state_dict = load(local_model_path, map_location=DEVICE)
                model.load_state_dict(state_dict)
                loaded_model_source = f"gs://{bucket_name}/{model_file}"
                print("Successfully loaded model from GCS")
            except Exception as e:
                print(f"Failed to load from GCS: {e}. Falling back to local model.")

    # Final fallback to base model
    if loaded_model_source == "None":
        print("Using fallback local base_model.pth")
        if os.path.exists("./models/base_model.pth"):
            state_dict = load("./models/base_model.pth", map_location=DEVICE)
            model.load_state_dict(state_dict)
            loaded_model_source = "Local base_model.pth"
        else:
            print("No model found! Using random weights.")
            loaded_model_source = "Random Weights (No model found)"

    transform = T.Compose(
        [
            T.Resize((32, 32)),
            T.ToTensor(),
        ]
    )

    print(f"Model loaded from: {loaded_model_source}")
    print(f"Model: {model}\n")
    print("Model ready for inference\n")

    yield
    print("Cleaning up...")
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
        "loaded_model_source": loaded_model_source,
    }


@app.get("/wandb-models")
def get_wandb_models(
    limit_collections: int = Query(5, ge=1, le=50, description="Max number of collections to scan"),
    latest_per_collection: bool = Query(True, description="Return only the newest version per collection"),
    per_collection: int = Query(
        1, ge=1, le=20, description="Max artifacts per collection if not latest_per_collection"
    ),
    limit_models: int = Query(10, ge=1, le=200, description="Max number of models returned"),
):
    """Returns available models from wandb registry"""
    try:
        api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
        entity = os.getenv("WANDB_ENTITY")
        project = os.getenv("WANDB_PROJECT")

        if not entity or not project:
            return {"error": "WANDB_ENTITY and WANDB_PROJECT must be set", "models": []}

        collections = api.artifact_collections(type_name="model", project_name=f"{entity}/{project}", per_page=5)

        collections = list(collections)
        total_collections = len(collections)
        print(f"Total collections found: {total_collections}")

        # Keep this endpoint fast: only scan a small number of collections.
        collections = collections[:limit_collections]

        models = []

        for coll in collections:
            # The W&B SDK may support pagination args; fall back gracefully.
            try:
                artifact_iter = coll.artifacts(per_page=1 if latest_per_collection else per_collection)
            except TypeError:
                artifact_iter = coll.artifacts()

            taken = 0
            for art in artifact_iter:
                models.append(
                    {
                        "collection_name": coll.name,
                        "version": art.version,
                        "full_path": f"{art.entity}/{art.project}/{coll.name}:{art.version}",
                        "created_at": (
                            art.created_at
                            if isinstance(art.created_at, str)
                            else art.created_at.isoformat()
                            if art.created_at
                            else None
                        ),
                        "aliases": list(art.aliases) if art.aliases else [],
                    }
                )
                taken += 1
                if latest_per_collection:
                    break
                if taken >= per_collection:
                    break

        # Sort newest first (ISO strings sort correctly)
        models.sort(key=lambda x: x["created_at"] or "", reverse=True)

        models = models[:limit_models]

        print(f"Returning {len(models)} models from wandb")
        return {
            "total_collections": total_collections,
            "scanned_collections": len(collections),
            "returned_models": len(models),
            "models": models,
        }

    except Exception as e:
        print(f"Error fetching wandb models: {e}")
        return {"error": str(e), "models": []}


@app.post("/switch-model")
def switch_model(model_path: str):
    """Switch to a different model from wandb registry

    Args:
        model_path: Full wandb artifact path (e.g., 'entity/project/model-name:version')
    """
    global model, loaded_model_source
    try:
        model = load_model_wandb(model_path).to(DEVICE)
        loaded_model_source = model_path
        return {
            "success": True,
            "message": f"Successfully switched to model: {model_path}",
            "loaded_model_source": loaded_model_source,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------------------------------ POST ------------------------------------------
@app.post("/model/")
async def model_inference(data: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Answers if the Image Sent is AI art of not

    Example Usage:
        curl -X POST "http://localhost:8000/model/" \
            -F "data=@cat.jpg" \
        """

    image_bytes = await data.read()
    img = await image_clean_utility(image_bytes)
    with no_grad():
        embeddings = model.classifier(model.backbone(img))
        logits = model.head(embeddings)
        prob = sigmoid(logits)
        is_ai = (prob > 0.5).item()

    background_tasks.add_task(
        add_to_database,
        latency=time.time(),
        embedding=embeddings.cpu().numpy(),
        prediction=prob.item(),
    )
    return {"isAI": is_ai, "probability": prob.item()}


async def image_clean_utility(image_bytes: bytes):
    i_image = Image.open(BytesIO(image_bytes))
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    image = cast(Tensor, transform(i_image))
    return cast(Tensor, unsqueeze(image, 0).float().to(DEVICE))
