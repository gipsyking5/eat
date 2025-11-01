FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# --- BASIC + TRAIN ---
RUN pip install --no-cache-dir \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh open3d xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- XFORMERS (voor PyTorch 2.4 + cu121) ---
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# --- FLASH-ATTN (prebuilt wheel) ---
RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu121torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# --- KAOLIN (voor PyTorch 2.4 + cu121) ---
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# --- NVDIFFRAST (from source) ---
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast

# --- DIFFOCTREERAST (from source) ---
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast

# --- MIP-SPLATTING (diff-gaussian-rasterization) ---
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# --- SPCONV (cu118) ---
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# --- TRELLIS ---
RUN pip install --no-cache-dir git+https://github.com/microsoft/TRELLIS.git

# --- API ---
COPY requirements.txt .
COPY app.py .
RUN pip install --no-cache-dir -r requirements.txt  # fastapi, uvicorn, etc.

RUN mkdir -p /workspace/outputs
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
