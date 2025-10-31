# app.py — EatTravelPlay TRELLIS GLB API (API-only, gebaseerd op originele c3-trellis-gradio)
import os
import uuid
import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- ALLE ORIGINELE IMPORTS BEHOUDEN ---
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from trellis.representations import Gaussian, MeshExtractResult

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- EXACT ZOALS IN ORIGINELE __main__ ---
model_repo = os.environ.get("TRELLIS_MODEL_REPO", "jetx/trellis-image-large")
print(f"Loading Trellis model from: {model_repo}")
pipeline = TrellisImageTo3DPipeline.from_pretrained(model_repo)
pipeline = pipeline.cuda()
pipeline = pipeline.half()  # FP16 voor RTX 4090

try:
    pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
    print("Rembg preloaded!")
except Exception as e:
    print(f"Preload failed: {e}")

print("TRELLIS model loaded!")

# Output dir
OUTPUT_DIR = "/workspace/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
async def root():
    return {"status": "TRELLIS GLB API ready"}

@app.post("/generate")
async def generate_glb(
    image: str = Form(...),
    prompt: str = Form("A delicious dish"),
    seed: int = Form(42),
    ss_guidance_strength: float = Form(7.5),
    ss_sampling_steps: int = Form(12),
    slat_guidance_strength: float = Form(3.0),
    slat_sampling_steps: int = Form(12)
):
    try:
        # Decode image
        img = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")

        # --- EXACT ZOALS IN ORIGINELE `image_to_3d()` ---
        outputs = pipeline.run(
            img,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )

        # --- EXACT ZOALS IN ORIGINELE `extract_glb()` ---
        gs = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]
        glb = postprocessing_utils.to_glb(
            gs, mesh,
            simplify=0.95,
            texture_size=1024,
            verbose=False
        )

        glb_id = str(uuid.uuid4())
        glb_path = f"{OUTPUT_DIR}/{glb_id}.glb"
        glb.export(glb_path)

        base_url = os.environ.get("PUBLIC_URL", "http://localhost:8000")
        model_url = f"{base_url}/outputs/{glb_id}.glb"

        return {"modelUrl": model_url}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
