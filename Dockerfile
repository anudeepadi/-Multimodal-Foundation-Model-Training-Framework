# Multi-stage Dockerfile for multimodal foundation model training
# Optimized for GPU training with CUDA support

# Base stage with CUDA runtime
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Development stage with full dependencies
FROM base as development

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    pre-commit \
    pytest-xdist \
    coverage

# Copy source code
COPY . .

# Install project in development mode
RUN pip install -e .

# Set up Jupyter
RUN jupyter nbextension enable --py widgetsnbextension

# Expose ports
EXPOSE 8888 6006 5000

# Default command for development
CMD ["bash"]

# Training stage - optimized for production training
FROM base as training

# Set working directory
WORKDIR /opt/multimodal-foundation-model

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY mlops/ ./mlops/
COPY setup.py .

# Install the project
RUN pip install -e .

# Create directories
RUN mkdir -p /data /outputs /models /logs

# Set environment variables
ENV PYTHONPATH=/opt/multimodal-foundation-model:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV OMP_NUM_THREADS=8

# Default training command
CMD ["python", "scripts/train_distributed.py", "--config", "configs/training_config.yaml"]

# Inference stage - lightweight for serving
FROM base as inference

WORKDIR /app

# Install minimal dependencies for inference
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy inference code
COPY src/models/ ./src/models/
COPY src/data/preprocessing.py ./src/data/preprocessing.py
COPY scripts/inference.py ./
COPY setup.py .

# Install minimal project
RUN pip install -e . --no-deps

# Create model directory
RUN mkdir -p /models

# Expose inference port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default inference command
CMD ["python", "inference.py", "--host", "0.0.0.0", "--port", "8000"]