FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Copy dependency files first (for faster builds)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Copy the source code
COPY src/ src/

# Create the data folder inside the container
RUN mkdir -p data/processed

# Run the script and tell it to save to /app/data/processed
ENTRYPOINT ["uv", "run", "python", "src/fakeartdetector/data.py", "data/processed"]