FROM ubuntu:24.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Install dependencies
COPY pyproject.toml ./
RUN uv sync --python 3.12

# Set the entrypoint to use the uv environment
ENV PATH="/app/.venv/bin:$PATH"

