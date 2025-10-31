#FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CUDA 12.8
RUN pip install --no-cache-dir torch==2.3.0+cu128 torchvision==0.18.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# TRELLIS + spconv
RUN pip install --no-cache-dir git+https://github.com/microsoft/TRELLIS.git
RUN pip install --no-cache-dir spconv-cu128

COPY app.py .

RUN mkdir -p /workspace/outputs
EXPOSE 8000

ENV PUBLIC_URL=http://localhost:8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
