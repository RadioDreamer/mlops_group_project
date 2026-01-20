# `Frontend (Streamlit)`

This project includes a Streamlit-based frontend used to upload images and request classification from the backend API.

## Purpose

- Provide an easy UI for model selection and inference.
- Allow downloading of inference logs and database.

## Running locally

Install project dependencies with the repository's package manager, then run:

```bash
uv sync
streamlit run src/fakeartdetector/frontend.py
```

## Environment variables

- `USE_LOCAL`: if set, the frontend will use `http://127.0.0.1:8000` as the backend.
- `BACKEND` or `BACKEND_URL`: override the backend URL (e.g., `https://example.com`).

## Behavior

- Fetches model list from the backend `/wandb-models` endpoint and shows latest models.
- Allows switching backend model via `/switch-model` when a model is selected.
- Uploads images to `/model/` for prediction.
- Lists and downloads inference logs from `/inference-logs/files` and `/download-db`.

## Deployment notes

- The frontend attempts to discover a Cloud Run service when not running locally. Configure `BACKEND` in production for reliability.
- Use Streamlit's `st.cache_resource` (already used) to cache long-lived objects like backend discovery and thread pools.

See the source: `src/fakeartdetector/frontend.py` for implementation details.
