# Base image with CUDA 12.1 and CuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# --- Critical Environment Variables ---
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models

# Set CUDA paths for compilation
ENV CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

# --- System Dependencies and Build Tools ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential git curl \
        libffi-dev libssl-dev zlib1g-dev \
        libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory for the application
WORKDIR /app

# --- PyTorch & Core GPU Libraries (CUDA 12.1) ---
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- Core Python Dependencies ---
RUN pip install --no-cache-dir \
    setuptools wheel huggingface_hub \
    pillow imageio imageio-ffmpeg tqdm easydict \
    scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- GPU Optimizations (C++/CUDA-heavy packages isolated) ---
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# --- Copy Local Project Files ---
# THESE LINES ARE CORRECT based on your GitHub images:
COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY requirements.txt .
COPY api_app.py .
COPY app.py .
# The problematic line COPY handler.py . has been removed!

# --- Install Project Extensions and Local Packages ---
# Install trellis as an editable package
RUN touch /app/trellis/__init__.py /app/trellis/pipelines/__init__.py && \
    pip install --no-cache-dir -e /app/trellis

# Install nvdiffrast extension
RUN pip install /app/extensions/nvdiffrast

# Install diffoctreerast (Source Clone and install)
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast && \
    rm -rf /tmp/extensions/diffoctreerast

# Install mip-splatting/diff-gaussian-rasterization (Source Clone and install)
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    rm -rf /tmp/extensions/mip-splatting

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

# Install server dependencies (uvicorn needed for CMD)
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart runpod

# Create necessary output directory (needed for the /workspace/outputs path in api_app.py)
RUN mkdir -p /workspace/outputs

# --- Command to start the FastAPI server ---
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8080"]
