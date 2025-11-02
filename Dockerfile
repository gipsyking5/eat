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

# --- BASIC + TRAIN ---
RUN pip install --no-cache-dir \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- XFORMERS ---
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# --- FLASH-ATTN (PYPI) ---
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- KAOLIN ---
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# --- NVDIFFRAST (source) ---
RUN mkdir -p /tmp/extensions && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast

# --- DIFFOCTREERAST (source) ---
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast

# --- MIPGAUSSIAN (diff-gaussian) ---
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# --- SPCONV ---
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# --- TRELLIS VAN HF (WHEEL, GEEN GIT) ---
RUN pip install --no-cache-dir \
    https://huggingface.co/jetx/trellis-image-large/resolve/main/trellis-0.0.1-py3-none-any.whl

# --- WHEELS ---
RUN pip install --no-cache-dir \
    https://huggingface.co/spaces/cavargas10/TRELLIS-Imagen3D/resolve/main/wheels/nvdiffrast-0.3.3-cp310-cp310-linux_x86_64.whl

# --- API DEPENDENCIES ---
# Added 'python-multipart' which is required for FastAPI to handle file uploads
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# --- JOUW BESTANDEN ---
# NOTE: Assuming you rename the new FastAPI file to 'api_app.py' to avoid confusion.
COPY api_app.py .
COPY requirements.txt .

RUN mkdir -p /workspace/outputs

EXPOSE 8000

# --- NEW CMD: Run FastAPI using Python entrypoint for better control and logging ---
CMD ["python3", "api_app.py"]
