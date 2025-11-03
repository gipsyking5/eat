# api_app.py (FINAL FULL CODE)
import os
import time
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import from your local trellis (now installed as package)
from trellis.pipelines import TrellisImageTo3DPipeline

# --- CRITICAL FIX: Load the model from the local repository path ---
# 1. Define the local path where your 'trellis' folder is copied inside the container.
LOCAL_MODEL_PATH = "/app/trellis"

# 2. Get the model source from the ENV variable, if set.
# If you set TRELLIS_MODEL_REPO="/app/trellis", it uses the local path.
# If you set TRELLIS_MODEL_REPO="microsoft/TRELLIS-image-large", it attempts to use that ID.
MODEL_SOURCE = os.environ.get("TRELLIS_MODEL_REPO", LOCAL_MODEL_PATH) 

try:
    # Load the pipeline using the configured source (should load locally if it's the path)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_SOURCE)
except Exception as e:
    print(f"Error loading model from {MODEL_SOURCE}. Falling back to base constructor. Error: {e}")
    # Fallback if from_pretrained requires a name or fails; assumes implicit loading of local files
    pipeline = TrellisImageTo3DPipeline() 

pipeline.cuda()
print(f"TRELLIS LOADED from source: {MODEL_SOURCE}")

app = FastAPI()

class Input(BaseModel):
    image: str
    prompt: str = "a 3D object"

def save_img(b64: str, name: str) -> str:
    if ',' in b64: b64 = b64.split(',')[1]
    # Use /tmp as it's safe for temporary files
    path = f"/tmp/{name}"
    with open(path, 'wb') as f:
        f.write(base64.b64decode(b64))
    return path

@app.post("/run")
async def run(data: Input):
    try:
        ts = int(time.time())
        img_path = save_img(data.image, f"in_{ts}.jpg")
        # Ensure /workspace/outputs exists, which it does via the Dockerfile
        out_dir = f"/workspace/outputs/job_{ts}" 
        os.makedirs(out_dir, exist_ok=True)
        glb_path = f"{out_dir}/model.glb"

        outputs = pipeline.run(img_path, prompt=data.prompt, seed=42)
        outputs['mesh'][0].save(glb_path)

        return {
            "status": "COMPLETED",
            "output": [{"name": glb_path, "mime": "model/gltf-binary"}]
        }
    except Exception as e:
        # Log the error internally and return a 500
        print(f"Runtime error in /run: {e}")
        raise HTTPException(500, str(e))

@app.get("/health")
def health(): return {"status": "OK"}
