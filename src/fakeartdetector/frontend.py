import io
import os
from concurrent.futures import Future, ThreadPoolExecutor
from time import sleep
from typing import cast

import pandas as pd
import requests
import streamlit as st
import wandb
from dotenv import load_dotenv
from google.cloud import run_v2

# Load environment variables from .env file
load_dotenv()


def get_wandb_models_from_api(
    backend: str,
    *,
    limit_collections: int = 5,
    latest_per_collection: bool = True,
    limit_models: int = 10,
):
    """Fetch available models from the backend API instead of wandb directly."""
    try:
        backend = (backend or "").strip().strip('"').strip("'")
        backend = backend.rstrip("/")
        url = f"{backend}/wandb-models"
        params = {
            "limit_collections": limit_collections,
            "latest_per_collection": str(latest_per_collection).lower(),
            "limit_models": limit_models,
        }
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"Successfully fetched {len(models)} models from backend API")
            return models

        # If the deployed backend doesn't have this endpoint yet, fall back to W&B.
        if response.status_code == 404:
            print("Backend does not expose /wandb-models (404). Falling back to W&B direct.")
            return get_wandb_models_direct(
                limit_collections=limit_collections,
                latest_per_collection=latest_per_collection,
                limit_models=limit_models,
            )

        print(f"API returned status code {response.status_code}")
        print(f"API returned msg {response.text} {response}")
        return []
    except Exception as e:
        print(f"Error fetching models from API: {e}")
        import traceback

        traceback.print_exc()
        # Final fallback
        try:
            return get_wandb_models_direct(
                limit_collections=limit_collections,
                latest_per_collection=latest_per_collection,
                limit_models=limit_models,
            )
        except Exception:
            return []


def get_wandb_models_direct(
    *,
    limit_collections: int = 5,
    latest_per_collection: bool = True,
    per_collection: int = 1,
    limit_models: int = 10,
) -> list[dict]:
    """Fallback: query W&B directly (still limited) when backend endpoint is unavailable."""
    api_key = os.getenv("WANDB_API_KEY")
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")
    if not api_key or not entity or not project:
        return []

    api = wandb.Api(api_key=api_key)
    collections = api.artifact_collections(type_name="model", project_name=f"{entity}/{project}")
    collections = list(collections)[:limit_collections]

    models: list[dict] = []
    for coll in collections:
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

    models = _sort_and_limit_models(models, limit=limit_models)
    return models


def _model_display_name(model: dict) -> str:
    name = model.get("name") or model.get("collection_name") or "(unknown)"
    version = model.get("version") or "?"
    aliases = model.get("aliases") or []
    alias_str = ", ".join(aliases) if aliases else "no alias"
    created_at = model.get("created_at")
    created_str = f" | {created_at}" if created_at else ""
    return f"{name}:{version} ({alias_str}){created_str}"


def _sort_and_limit_models(models: list[dict], limit: int = 10) -> list[dict]:
    def sort_key(m: dict) -> str:
        # Backend already returns ISO strings; lexical sort works.
        return m.get("created_at") or ""

    models_sorted = sorted(models, key=sort_key, reverse=True)
    return models_sorted[:limit]


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    # Check environment variable first
    # return "http://127.0.0.1:8000"

    env_backend = os.environ.get("BACKEND") or os.environ.get("BACKEND_URL")
    if env_backend:
        env_backend = env_backend.strip().strip('"').strip("'").rstrip("/")
        print(f"Using BACKEND from .env: {env_backend}")
        return env_backend

    # # Try to discover Cloud Run service automatically
    try:
        parent = "projects/double-zenith-484209-d9/locations/europe-west1"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        print("Discovered Cloud Run services:")
        for service in services:
            service_name = service.name.split("/")[-1]
            print(f"  - {service_name}: {service.uri}")
            if service_name == "fake-art-api":
                discovered_url = str(service.uri)  # Convert to string
                print(f"Using discovered service: {discovered_url}")
                return discovered_url
    except Exception as e:
        print(f"Could not discover Cloud Run service: {e}")

    # Fallback for local development
    return None


@st.cache_resource
def _models_executor() -> ThreadPoolExecutor:
    # Single worker is enough; keeps requests serialized and avoids hammering the backend.
    return ThreadPoolExecutor(max_workers=1)


def _start_models_fetch(backend: str) -> Future[list[dict]]:
    executor = _models_executor()
    return executor.submit(
        get_wandb_models_from_api,
        backend,
        limit_collections=5,
        latest_per_collection=True,
        limit_models=10,
    )


def _rerun() -> None:
    # Streamlit renamed this API over time.
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def classify_image(image: bytes, backend: str, filename: str | None = None, mime: str | None = None):
    """Send the image to the backend for classification.

    The API expects a multipart-form field named "data" (UploadFile).
    """
    predict_url = f"{backend}/model/"
    file_tuple = (filename or "image", io.BytesIO(image), mime or "application/octet-stream")
    response = requests.post(predict_url, files={"data": file_tuple}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def switch_model_on_backend(backend: str, model_path: str):
    """Tell the backend to switch to a different model."""
    switch_url = f"{backend}/switch-model"
    params = {"model_path": model_path}
    try:
        response = requests.post(switch_url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    # Add model selection in sidebar
    st.sidebar.title("Model Selection")
    st.sidebar.caption(f"Backend: {backend}")

    # Fetch models in the background so the UI renders fast.
    if "models_future" not in st.session_state:
        st.session_state.models_future = _start_models_fetch(backend)
        st.session_state.models_data = None
        st.session_state.models_error = None
        st.session_state.models_poll_count = 0

    future: Future[list[dict]] | None = st.session_state.get("models_future")
    if future is not None and future.done() and st.session_state.get("models_data") is None:
        try:
            st.session_state.models_data = future.result()
        except Exception as e:
            st.session_state.models_error = str(e)
            st.session_state.models_data = []

    if st.sidebar.button("Refresh models"):
        st.session_state.models_future = _start_models_fetch(backend)
        st.session_state.models_data = None
        st.session_state.models_error = None
        st.session_state.models_poll_count = 0
        _rerun()

    if st.session_state.get("models_data") is None:
        st.sidebar.info("Loading models…")
        # Auto-rerun a few times while waiting for the background fetch to complete.
        # Without this, Streamlit may not rerun at the moment the future completes,
        # and the sidebar will look like it's stuck.
        poll_count = int(st.session_state.get("models_poll_count", 0) or 0)
        if poll_count < 30:
            st.session_state.models_poll_count = poll_count + 1
            sleep(0.4)
            _rerun()
        return
    else:
        available_models = cast(list[dict], st.session_state.get("models_data") or [])
        available_models = _sort_and_limit_models(available_models, limit=10)

    if st.session_state.get("models_error"):
        st.sidebar.warning(f"Model list failed: {st.session_state.models_error}")

    if available_models:
        st.sidebar.caption(f"Showing {len(available_models)} latest models")

        selected_model = st.sidebar.selectbox(
            "Choose a model:",
            options=[None] + available_models,
            format_func=lambda m: "Default Model" if m is None else _model_display_name(m),
            help="Select which model to use for prediction",
        )

        if selected_model is not None:
            selected_model_path = selected_model.get("full_path")
            if selected_model_path:
                st.sidebar.info(f"Selected: {selected_model_path}")

                if st.sidebar.button("Switch to this model"):
                    with st.spinner("Switching model on backend..."):
                        result = switch_model_on_backend(backend, selected_model_path)
                        if result.get("success"):
                            st.sidebar.success(f"✅ {result.get('message', 'Model switched successfully!')}")
                        else:
                            st.sidebar.error(f"❌ Failed to switch model: {result.get('error', 'Unknown error')}")
            else:
                st.sidebar.error("Selected model is missing 'full_path' from the API")
        else:
            selected_model_path = None
            st.sidebar.info("Using default model from environment")
    else:
        st.sidebar.warning("No models found (showing default model).")
        selected_model_path = None

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()

        result = classify_image(
            image,
            backend=backend,
            filename=getattr(uploaded_file, "name", None),
            mime=getattr(uploaded_file, "type", None),
        )

        if result is not None:
            is_ai = result.get("isAI")
            prob_ai = float(result.get("probability", 0.0))

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", "AI" if is_ai else "Human")

            # two-class probability bar chart (AI vs Human)
            data = {"Class": ["AI", "Human"], "Probability": [prob_ai, max(0.0, 1.0 - prob_ai)]}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
