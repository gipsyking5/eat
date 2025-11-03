FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0" \
    HF_HOME=/app/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        build-essential git curl \
        libffi-dev libssl-dev zlib1g-dev \
        libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    setuptools wheel huggingface_hub \
    pillow imageio imageio-ffmpeg tqdm easydict \
    opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN pip install --no-cache-dir spconv-cu118==2.3.6

COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY requirements.txt .
COPY api_app.py .
COPY handler.py .

# FIX IMPORT: Add __init__.py + install trellis as package (no model download here)
RUN touch /app/trellis/__init__.py /app/trellis/pipelines/__init__.py && \
    pip install --no-cache-dir -e /app/trellis

RUN pip install /app/extensions/nvdiffrast

RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && \
    pip install /tmp/diffoctreerast && rm -rf /tmp/diffoctreerast

RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && \
    pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ && \
    rm -rf /tmp/mip-splatting

RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart runpod

RUN mkdir -p /workspace/outputs

CMD ["python", "-m", "runpod", "start", "--handler", "handler.handler"]
