# Use a base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential packages, including git
RUN apt-get update && \
    apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# --- PYTORCH & CORE LIBRARIES (CUDA 12.1) ---
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- CORE PYTHON DEPENDENCIES ---
# Includes necessary non-trellis specific libraries
RUN pip install --no-cache-dir \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- GPU OPTIMIZATIONS (XFORMERS, FLASH-ATTN, KAOLIN) ---
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- KAOLIN ---
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# --- SPCONV (Requires CUDA) ---
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# --- COPY LOCAL PROJECT FILES ---
# This is where your forked 'trellis' code and local dependencies are copied in.
COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY requirements.txt .
# Copy the new FastAPI server code
COPY api_app.py .

# --- INSTALL TRELLIS DEPENDENCIES FROM LOCAL SOURCE ---
# 1. nvdiffrast (Install from the local source folder copy)
RUN pip install /app/extensions/nvdiffrast

# 2. DIFFOCTREERAST (Source Clone)
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast

# 3. MIPGAUSSIAN (Source Clone)
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# --- API DEPENDENCIES ---
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

RUN mkdir -p /workspace/outputs

EXPOSE 8000

# --- FINAL COMMAND: Runs the FastAPI server using Uvicorn's direct command ---
# It serves the 'app' object inside the 'api_app' module on 0.0.0.0:8000
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
