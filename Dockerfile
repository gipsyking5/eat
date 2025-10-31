FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir git+https://github.com/microsoft/TRELLIS.git

COPY app.py .

RUN mkdir -p /workspace/outputs
EXPOSE 8000

ENV PUBLIC_URL=http://localhost:8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
