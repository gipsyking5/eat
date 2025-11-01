FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# --- SYSTEM DEPS ---
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# --- TORCH + CUDA 12.1 ---
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- BASIC + TRAIN (uit setup.sh) ---
RUN pip install --no-cache-dir \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- XFORMERS (uit setup.sh) ---
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# --- FLASH-ATTN (uit setup.sh — VAN PYPI, GEEN 404) ---
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- KAOLIN (uit setup.sh) ---
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# --- NVDIFFRAST (uit setup.sh — van source) ---
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast

# --- DIFFOCTREERAST (uit setup.sh — van source) ---
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast

# --- MIPGAUSSIAN (uit setup.sh — diff-gaussian-rasterization) ---
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# --- SPCONV (uit setup.sh — cu118) ---
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# --- TRELLIS ---
RUN pip install --no-cache-dir git+https://github.com/microsoft/TRELLIS.git

# --- WHEELS (alleen nvdiffrast + diff-gaussian) ---
RUN pip install --no-cache-dir \
    https://huggingface.co/spaces/cavargas10/TRELLIS-Imagen3D/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl

# --- API DEPS ---
RUN pip install --no-cache-dir fastapi uvicorn numpy

# --- JOUW BESTANDEN ---
COPY app.py .
COPY requirements.txt .

RUN mkdir -p /workspace/outputs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
