# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

COPY main.py .
COPY scripts/ ./scripts/

ENV TORCH_HOME /workspace/models


VOLUME /workspace/video_data
VOLUME /workspace/video_cluster
VOLUME /workspace/hf_cache
VOLUME /workspace/models

CMD ["python", "main.py"]