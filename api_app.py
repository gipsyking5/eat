# api_app.py
import os
import time
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# Import from installed trellis package
from trellis.pipelines import TrellisImageTo3DPipeline

# Load once at startup
print("Loading TRELLIS model...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
print("TRELLIS model loaded on GPU!")

app = FastAPI()

class TrellisInput(BaseModel):
    image: str  # data:image/...;base64,...
    prompt: str = "a 3D scanned object"

def save_base64_image(b64: str, filename: str) -> str:
    if ',' in b64:
        b64 = b64.split(',')[1]
    data = base64.b64decode(b64)
    path = f"/tmp/{filename}"
    with open(path, 'wb') as f:
        f.write(data)
    return path

@app.post("/run")
async def run_pipeline(data: TrellisInput):
    try:
        ts = int(time.time())
        img_path = save_base64_image(data.image, f"input_{ts}.jpg")

        print(f"Generating 3D for: {data.prompt}")
        outputs = pipeline.run(img_path, prompt=data.prompt, seed=42)
        mesh = outputs['mesh'][0]

        out_dir = f"/workspace/outputs/job_{ts}"
        os.makedirs(out_dir, exist_ok=True)
        glb_path = f"{out_dir}/model.glb"
        mesh.save(glb_path)

        return {
            "status": "COMPLETED",
            "output": [
                {"name": glb_path, "mime": "model/gltf-binary"}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "RUNNING", "pipeline": True}
