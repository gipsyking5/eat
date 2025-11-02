# --------------------------------------------------------------
#  RunPod-Ready Dockerfile: CUDA 12.1 + PyTorch 2.4 + Trellis Pipeline
#  Fixes: Git layer, trellis imports, heavy compiles
# --------------------------------------------------------------

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Environment setup (non-interactive, CUDA paths, Python unbuffered)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

# -----------------------------------------------------------------
# 1. Install system dependencies (git, build tools, libs)
# -----------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential git curl \
        libffi-dev libssl-dev zlib1g-dev \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------
# 2. Core PyTorch (CUDA 12.1)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------
# 3. Core Python dependencies (safe, non-compiling first)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    setuptools wheel \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# -----------------------------------------------------------------
# 4. GPU-heavy optimizations (compile after basics)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# -----------------------------------------------------------------
# 5. Copy local project files (your trellis source)
# -----------------------------------------------------------------
COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY requirements.txt .
COPY api_app.py .

# -----------------------------------------------------------------
# 6. CRITICAL FIX: Install local 'trellis' as editable package
#    (Makes 'from trellis.pipelines import TrellisImageTo3DPipeline' work at runtime)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir -e /app/trellis

# -----------------------------------------------------------------
# 7. Install local extensions
# -----------------------------------------------------------------
RUN pip install /app/extensions/nvdiffrast

# -----------------------------------------------------------------
# 8. Clone & install external extensions (git in SAME layer each time)
# -----------------------------------------------------------------
# 8.1 DIFFOCTREERAST
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast && \
    apt-get purge -y --auto-remove git && \
    rm -rf /var/lib/apt/lists/* /tmp/extensions/diffoctreerast

# 8.2 MIP-SPLATTING (diff-gaussian-rasterization)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    apt-get purge -y --auto-remove git && \
    rm -rf /var/lib/apt/lists/* /tmp/extensions/mip-splatting

# -----------------------------------------------------------------
# 9. Install from requirements.txt + API deps (safe fallback)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# Create outputs dir
RUN mkdir -p /workspace/outputs

# Expose port
EXPOSE 8000

# -----------------------------------------------------------------
# 10. Start the FastAPI app
# -----------------------------------------------------------------
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
