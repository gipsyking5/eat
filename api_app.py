# api_app.py
import os
import time
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import from your local trellis (now installed as package)
from trellis.pipelines import TrellisImageTo3DPipeline

# Load once
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
print("TRELLIS LOADED")

app = FastAPI()

class Input(BaseModel):
    image: str
    prompt: str = "a 3D object"

def save_img(b64: str, name: str) -> str:
    if ',' in b64: b64 = b64.split(',')[1]
    path = f"/tmp/{name}"
    with open(path, 'wb') as f:
        f.write(base64.b64decode(b64))
    return path

@app.post("/run")
async def run(data: Input):
    try:
        ts = int(time.time())
        img_path = save_img(data.image, f"in_{ts}.jpg")
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
        raise HTTPException(500, str(e))

@app.get("/health")
def health(): return {"status": "OK"}
