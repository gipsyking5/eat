# Use a base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- CRITICAL FIX 1: Set explicit CUDA paths and dependencies ---
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# --- CRITICAL FIX 2: Manually set CUDA architecture list to fix the IndexError ---
# This tells the compiler which GPUs to target (e.g., V100/A100/A6000 compatibility)
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

# Install ALL system dependencies for compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential git curl \
    libffi-dev libssl-dev zlib1g-dev \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch first
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- CORE PYTHON DEPENDENCIES (Safe/Non-Compiling packages first) ---
RUN pip install --no-cache-dir \
    setuptools wheel \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- GPU OPTIMIZATIONS (C++/CUDA-heavy packages installed LAST) ---
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN pip install --no-cache-dir spconv-cu118==2.3.6
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- COPY LOCAL PROJECT FILES ---
COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY requirements.txt .
COPY api_app.py .

# --- INSTALL TRELLIS DEPENDENCIES FROM LOCAL SOURCE (The specific fix is the manual python setup.py) ---

# 1. nvdiffrast (Should now compile with fixed CUDA architecture flags)
RUN cd /app/extensions/nvdiffrast && \
    python3 setup.py install

# 2. DIFFOCTREERAST (Source Clone and install)
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast

# 3. MIPGAUSSIAN (Source Clone and install)
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# --- API DEPENDENCIES ---
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

RUN mkdir -p /workspace/outputs

EXPOSE 8000

# --- FINAL COMMAND: Runs the FastAPI server using Uvicorn's direct command ---
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
