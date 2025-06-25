# QwenVL Medical Inference Docker Image
# FLARE 2025 Medical Multimodal VQA Challenge
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for optimal inference
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# PyTorch is already installed in the base image, so we skip that step

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download the FLARE 2025 fine-tuned model during build
COPY download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py && rm /tmp/download_models.py

# Create application directory
WORKDIR /app

# Copy inference script and related files
COPY inference.py /app/
COPY docker_inference.sh /app/

# Create necessary directories and handle optional utils
RUN mkdir -p /app/input /app/output /app/models /app/logs /app/utils

# Make scripts executable
RUN chmod +x /app/docker_inference.sh

# Set working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["./docker_inference.sh"] 