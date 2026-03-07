# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Copy uv binaries from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use uv to install Python packages quickly into the system Python
RUN uv pip install --system --no-cache \
    opencv-python-headless \
    scikit-learn \
    torchvision \
    datasets \
    pandas \
    plotly

    
# Copy your script into the container
COPY main.py .

# Default command
CMD ["python", "main.py"]