# --------------------------------------------------------------
#  RunPod-Ready Dockerfile: Full TRELLIS + CUDA 12.1 + API
#  Fixes: git warning, TrellisImageTo3DPipeline import, model download
# --------------------------------------------------------------

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0" \
    HF_HOME=/app/models

# -----------------------------------------------------------------
# 1. System dependencies + git (fixes "git not found" warning)
# -----------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential git curl \
        libffi-dev libssl-dev zlib1g-dev \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------
# 2. Core PyTorch (CUDA 12.1)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------
# 3. Core Python dependencies
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    setuptools wheel huggingface_hub \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# -----------------------------------------------------------------
# 4. GPU-heavy optimizations
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# -----------------------------------------------------------------
# 5. FULL OFFICIAL TRELLIS: Clone + package + model download
# -----------------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /app/trellis && \
    # Ensure importable as package
    touch /app/trellis/__init__.py && \
    touch /app/trellis/pipelines/__init__.py && \
    # Install as editable package
    pip install --no-cache-dir -e /app/trellis && \
    # Pre-download model (cached in /app/models)
    python3 - <<'PY'
from trellis.pipelines import TrellisImageTo3DPipeline
print("Downloading TRELLIS model...")
TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
print("Model downloaded and cached!")
PY

# -----------------------------------------------------------------
# 6. Your custom extensions
# -----------------------------------------------------------------
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
RUN pip install /app/extensions/nvdiffrast

# 6.1 DIFFOCTREERAST
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
    pip install /tmp/diffoctreerast && \
    rm -rf /tmp/diffoctreerast

# 6.2 MIP-SPLATTING
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    rm -rf /tmp/mip-splatting

# -----------------------------------------------------------------
# 7. Your API code
# -----------------------------------------------------------------
COPY requirements.txt .
COPY api_app.py .

# -----------------------------------------------------------------
# 8. Install API dependencies
# -----------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# -----------------------------------------------------------------
# 9. Outputs directory
# -----------------------------------------------------------------
RUN mkdir -p /workspace/outputs

EXPOSE 8000

# -----------------------------------------------------------------
# 10. Start FastAPI
# -----------------------------------------------------------------
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
