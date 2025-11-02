# --- BASE IMAGE & ENVIRONMENT SETUP ---
# Use a base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- CRITICAL FIX 1: CUDA PATH AND ARCHITECTURE ---
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# CRITICAL FIX: Manually set CUDA architecture list (Fixes 'list index out of range' error)
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

# --- SYSTEM DEPENDENCIES ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential git curl \
    libffi-dev libssl-dev zlib1g-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# --- CORE PYTHON & PYTORCH ---
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- CRITICAL FIX 2: CLONE, INSTALL, AND RESOLVE DEPENDENCIES ---
# We assume the Trellis code is inside a repository that needs cloning.
RUN git clone https://github.com/Trellis-App/trellis-pipeline.git /app/trellis_repo \
    && cd /app/trellis_repo \
    && git submodule update --init --recursive \
    && pip install /app/trellis_repo

# Install other dependencies
RUN pip install --no-cache-dir \
    setuptools wheel \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html \
    spconv-cu118==2.3.6 \
    xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 \
    flash-attn --no-build-isolation

# --- API SETUP ---
COPY api_app.py .

RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

RUN mkdir -p /workspace/outputs

EXPOSE 8000

# --- FINAL COMMAND ---
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
