FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# installing system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# setup the workspace
WORKDIR /app

# copying dependencies files first
COPY uv.lock pyproject.toml ./

# installing dependencies
RUN uv sync --frozen --no-cache --no-install-project

# copyng the rest of the project
COPY src/ src/
COPY models/ models/
COPY configs/ configs/
COPY README.md README.md

# syncing
RUN uv sync --frozen
EXPOSE $PORT
ENTRYPOINT ["uv", "run", "uvicorn", "src.fakeartdetector.api:app", "--host", "0.0.0.0", "--port", "$PORT"]
