# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dependencies for OpenCV and other graphics/compute libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    # Utilities to download the static ffmpeg build
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    # Download and install a full-featured static ffmpeg build that includes libx264
    && FFMPEG_VERSION="7.0" \
    && wget "https://johnvansickle.com/ffmpeg/releases/ffmpeg-${FFMPEG_VERSION}-amd64-static.tar.xz" \
    && tar -xvf "ffmpeg-${FFMPEG_VERSION}-amd64-static.tar.xz" \
    #Move ffmpeg directly into Conda's bin folder to overwrite the bad version
    && mv "ffmpeg-${FFMPEG_VERSION}-amd64-static/ffmpeg" "ffmpeg-${FFMPEG_VERSION}-amd64-static/ffprobe" /opt/conda/bin/ \
    && rm -rf "ffmpeg-${FFMPEG_VERSION}-amd64-static.tar.xz" "ffmpeg-${FFMPEG_VERSION}-amd64-static"

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