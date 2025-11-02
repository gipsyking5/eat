@@ -17,60 +17,49 @@ WORKDIR /app
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- BASIC + TRAIN ---
# Note: Keep core dependencies outside of TRELLIS to ensure they are installed first.
# --- CORE PYTHON DEPENDENCIES ---
# Includes necessary non-trellis specific libraries
RUN pip install --no-cache-dir \
    pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime \
    trimesh xatlas pyvista pymeshfix igraph transformers \
    tensorboard pandas lpips \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# --- XFORMERS & FLASH-ATTN (GPU Optimizations) ---
# --- GPU OPTIMIZATIONS (XFORMERS, FLASH-ATTN, KAOLIN) ---
RUN pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# --- KAOLIN ---
RUN pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# --- SPCONV ---
# --- SPCONV (Requires CUDA) ---
RUN pip install --no-cache-dir spconv-cu118==2.3.6

# --- COPY PROJECT FILES FIRST (Including 'trellis' folder) ---
# Copy the necessary local source code directories
# --- COPY LOCAL PROJECT FILES ---
# This is where your forked 'trellis' code and local dependencies are copied in.
COPY trellis /app/trellis
COPY extensions/nvdiffrast /app/extensions/nvdiffrast
COPY wheels /app/wheels
COPY requirements.txt .
# Copy the new FastAPI server code
COPY api_app.py .

# --- INSTALL TRELLIS COMPONENTS FROM LOCAL SOURCE ---
# This is cleaner than using the remote wheel, as you have the source.
# --- INSTALL TRELLIS DEPENDENCIES ---
# 1. nvdiffrast (Install from the local source folder copy)
RUN pip install /app/extensions/nvdiffrast

# --- DIFFOCTREERAST (source) ---
# NOTE: This step must remain as a clone since the source isn't in your screenshot.
# 2. DIFFOCTREERAST (Source Clone)
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    pip install /tmp/extensions/diffoctreerast

# --- MIPGAUSSIAN (diff-gaussian) ---
# NOTE: This step must remain as a clone since the source isn't in your screenshot.
# 3. MIPGAUSSIAN (Source Clone)
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/


# --- API DEPENDENCIES ---
# Added FastAPI and file upload support
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# --- CLEANUP (Removed redundant TRELLIS and nvdiffrast wheel installs) ---
# Removed: TRELLIS VAN HF (WHEEL)
# Removed: WHEELS (nvdiffrast wheel)

RUN mkdir -p /workspace/outputs

EXPOSE 8000

# --- FINAL COMMAND: Run the new FastAPI application ---
# Make sure your server file is named 'api_app.py' as provided in the previous turn.
# --- FINAL COMMAND: Runs the FastAPI server using the provided script ---
CMD ["python3", "api_app.py"]

