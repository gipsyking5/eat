import os
import shutil
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Literal
from easydict import EasyDict as edict
import imageio

# FastAPI Imports for API endpoint
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware # Important for client apps

# TRELLIS Imports (from your original code)
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils

# --- Configuration and Initialization ---
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="TRELLIS 3D Generation API",
    description="A simple API for Image-to-3D conversion using the TRELLIS pipeline."
)

# Crucial for allowing your client app (React Native/Web) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be specific in production, e.g., ["https://your.client.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline initialization (TRELLIS model loading)
# This is done once when the server starts, reusing your original initialization logic.
try:
    model_repo = os.environ.get("TRELLIS_MODEL_REPO", "jetx/trellis-image-large")
    print(f"Loading Trellis model from: {model_repo}")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_repo)
    pipeline.cuda()
    # Preload rembg (as in your original code)
    pipeline.preprocess_image(Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)))
except Exception as e:
    print(f"Failed to load TRELLIS pipeline: {e}")
    pipeline = None # Set to None if loading fails

# --- Utility Functions from your original Gradio App ---

# Reusing the utility function to get the seed
def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

# Reusing the preprocessing function
def preprocess_image(image: Image.Image) -> Image.Image:
    if not pipeline:
        raise RuntimeError("TRELLIS pipeline is not loaded.")
    processed_image = pipeline.preprocess_image(image)
    return processed_image


# --- The Main API Endpoint ---

@app.post("/generate-3d/")
async def generate_3d_model_api(
    image_file: UploadFile = File(..., description="The input image file."),
    # Reusing generation settings as Form fields
    seed: int = Form(0, description="Seed for generation."),
    randomize_seed: bool = Form(True, description="Whether to randomize the seed."),
    ss_guidance_strength: float = Form(7.5, description="Guidance for Structure Stage."),
    ss_sampling_steps: int = Form(12, description="Steps for Structure Stage."),
    slat_guidance_strength: float = Form(3.0, description="Guidance for Detail Stage."),
    slat_sampling_steps: int = Form(12, description="Steps for Detail Stage."),
    mesh_simplify: float = Form(0.95, description="Mesh simplification ratio."),
    texture_size: int = Form(1024, description="Texture size for GLB export."),
    # Multi-image parameters are ignored for simplicity in this single-endpoint example
):
    """
    Receives an image and generation parameters, runs the TRELLIS pipeline, 
    and returns the generated GLB file directly.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="3D Pipeline not initialized. Check server logs.")

    if not image_file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # 1. Setup temporary directories and files
    # Note: In a true serverless environment, use request ID for unique folder names
    temp_id = os.urandom(16).hex()
    user_dir = os.path.join(TMP_DIR, temp_id)
    os.makedirs(user_dir, exist_ok=True)
    input_image_path = os.path.join(user_dir, "input_image.png")
    output_glb_path = os.path.join(user_dir, "generated_model.glb")

    try:
        # 2. Save and open the uploaded image
        with open(input_image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        
        input_image = Image.open(input_image_path)
        
        # 3. Preprocess the image (background removal, etc.)
        processed_image = preprocess_image(input_image)

        # 4. Generate 3D Model (equivalent to your image_to_3d function)
        final_seed = get_seed(randomize_seed, seed)

        outputs = pipeline.run(
            processed_image,
            seed=final_seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False, # Already preprocessed
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
        
        # 5. Extract Gaussian and Mesh
        gs = outputs['gaussian'][0]
        mesh = outputs['mesh'][0]

        # 6. Extract GLB (equivalent to your extract_glb function)
        glb = postprocessing_utils.to_glb(
            gs, 
            mesh, 
            simplify=mesh_simplify, 
            texture_size=texture_size, 
            verbose=False
        )
        glb.export(output_glb_path)
        
        # 7. Clean up CUDA memory
        torch.cuda.empty_cache()

        # 8. Return the GLB file
        # FileResponse will stream the file and clean up the temporary folder in the 'finally' block
        return FileResponse(
            path=output_glb_path,
            filename="trellis_model.glb",
            media_type='model/gltf-binary',
            # Add a header to force download/save name if needed
            headers={"Content-Disposition": "attachment; filename=trellis_model.glb"}
        )

    except Exception as e:
        print(f"An error occurred during 3D generation: {e}")
        # Return a 500 status on internal error
        raise HTTPException(status_code=500, detail=f"3D Model generation failed due to an internal error: {str(e)}")
        
    finally:
        # 9. Cleanup temporary folder (Crucial for serverless environments)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)

# --- Server Startup Command ---

# If you need to run this script directly (e.g., in a Docker CMD):
if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is mandatory for Docker/Serverless container hosting
    # We use 8000 as a standard API port
    uvicorn.run(app, host="0.0.0.0", port=8000)
