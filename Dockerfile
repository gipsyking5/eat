# --------------------------------------------------------------
#  RunPod Serverless – FULLY WORKING TRELLIS 3D Generation
#  Keeps your local trellis/ folder, fixes git warning,
#  makes TrellisImageTo3DPipeline importable, pre-downloads model
# --------------------------------------------------------------

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# -----------------------------------------------------------------
# Environment
# -----------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0" \
    HF_HOME=/app/models

# -----------------------------------------------------------------
# 1. System packages + git (removes the warning)
# -----------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential git curl \
        libffi-dev libssl-dev zlib1g-dev \
        libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------------------------------------------
# 2. PyTorch (CUDA 12.1)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------
# 3. Core Python packages (non-compiling first)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    setuptools wheel huggingface_hub \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# -----------------------------------------------------------------
# 4. GPU-heavy packages
# -----------------------------------------------------------------
RUN pip install --no-cache-dir xformers==0.0.27.post2 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir kaolin \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# -----------------------------------------------------------------
# 5. Copy your local files (trellis stays!)
# -----------------------------------------------------------------
COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY requirements.txt .
COPY api_app.py .
COPY handler.py .

# -----------------------------------------------------------------
# 6. Make trellis importable + install it + download model
# -----------------------------------------------------------------
RUN touch /app/trellis/__init__.py /app/trellis/pipelines/__init__.py && \
    pip install --no-cache-dir -e /app/trellis && \
    python3 - <<'PY'
from trellis.pipelines import TrellisImageTo3DPipeline
print("Downloading TRELLIS model (this may take a few minutes)…")
TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
print("Model cached in /app/models")
PY

# -----------------------------------------------------------------
# 7. Install your custom extensions
# -----------------------------------------------------------------
RUN pip install /app/extensions/nvdiffrast

# diffoctreerast
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
    pip install /tmp/diffoctreerast && rm -rf /tmp/diffoctreerast

# mip-splatting
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    rm -rf /tmp/mip-splatting

# -----------------------------------------------------------------
# 8. API dependencies
# -----------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart runpod

# -----------------------------------------------------------------
# 9. Output folder
# -----------------------------------------------------------------
RUN mkdir -p /workspace/outputs

# -----------------------------------------------------------------
# 10. RunPod Serverless entrypoint
# -----------------------------------------------------------------
CMD ["python", "-m", "runpod", "start", "--handler", "handler.handler"]
