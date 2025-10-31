FROM runpod/pytorch:2.4.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir git+https://github.com/microsoft/TRELLIS.git

RUN mkdir -p /workspace/outputs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
