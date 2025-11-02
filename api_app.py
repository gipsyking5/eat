import os
import io
import time
import base64
import uuid
import tempfile
import numpy as np
from PIL import Image

# Import necessary dependencies for the web server
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import uvicorn

# Import TRELLIS components (assumes local import from the 'trellis' folder in your repo)
try:
    from trellis.pipelines.image_to_3d import TrellisImageTo3DPipeline
except ImportError:
    # If the local import fails (e.g., during testing outside the container),
    # provide a helpful error. This shouldn't happen inside the final Docker container.
    print("FATAL: Could not import TrellisImageTo3DPipeline. Ensure 'trellis' folder is present and installed.")
    exit(1)

# --- CONFIGURATION ---

# IMPORTANT: The model weights are large and must be downloaded.
# The pipeline will use this HuggingFace repo ID to fetch the weights.
# This variable must be set in your RunPod environment.
MODEL_REPO = os.environ.get("TRELLIS_MODEL_REPO", "jetx/trellis-image-large")

# --- APP SETUP AND MODEL LOADING ---

app = FastAPI(
    title="TRELLIS Image-to-3D API",
    description="Minimal FastAPI endpoint for TRELLIS to convert an image to a GLB model."
)

# Global variable to hold the loaded pipeline
pipe = None

@app.on_event("startup")
async def startup_event():
    """
    Loads the TRELLIS model pipeline on application startup.
    This ensures the model is ready in memory for fast inference.
    """
    global pipe
    print(f"[{time.ctime()}] Starting up... Loading TRELLIS model from: {MODEL_REPO}")
    try:
        # Load the pipeline, enabling CUDA (assuming GPU is available in container)
        pipe = TrellisImageTo3DPipeline.from_pretrained(
            MODEL_REPO,
            device='cuda:0',
            trust_remote_code=True
        )
        print(f"[{time.ctime()}] TRELLIS Model loaded successfully.")
    except Exception as e:
        print(f"[{time.ctime()}] ERROR loading TRELLIS model: {e}")
        # Re-raise or exit if model loading is critical
        raise e

# --- UTILITY FUNCTION ---

def base64_to_pil_image(base64_string: str) -> Image.Image:
    """Converts a Base64 string (data URI or raw Base64) to a PIL Image object."""
    # Clean up common data URI prefixes if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


# --- API ENDPOINT ---

@app.post("/generate-3d/")
async def generate_3d(
    # NOTE: The client side (runpodService.ts) must send the image as a multipart file upload.
    image_file: UploadFile = File(..., description="Source image file (PNG/JPEG)"),
    prompt: str = Form("a delicious scanned dish", description="Text prompt to guide the 3D generation.")
):
    """
    Converts a single input image and prompt into a 3D GLB model.
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load.")

    try:
        # 1. Read Image and Convert to PIL
        image_bytes = await image_file.read()
        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Run Inference
        print(f"[{time.ctime()}] Starting generation for prompt: '{prompt}'...")

        # Create temporary file paths for the model output
        temp_dir = tempfile.gettempdir()
        output_glb_path = os.path.join(temp_dir, f"trellis_output_{uuid.uuid4()}.glb")

        # The TRELLIS pipeline function
        pipe(
            prompt=prompt,
            image=init_image,
            output_path=output_glb_path,
            # Additional settings you might need:
            num_steps=50, # Example step count
        )

        print(f"[{time.ctime()}] Generation complete. Model saved to {output_glb_path}")

        # 3. Read GLB Output
        with open(output_glb_path, 'rb') as f:
            glb_data = f.read()

        # 4. Clean up the temporary file
        os.remove(output_glb_path)

        # 5. Return GLB as binary response
        # The client (runpodService.ts) will handle this binary data to create a URL.
        return Response(content=glb_data, media_type="model/gltf-binary", headers={
            "Content-Disposition": f"attachment; filename=\"{prompt.replace(' ', '_')}.glb\"",
            "Access-Control-Expose-Headers": "Content-Disposition"
        })

    except Exception as e:
        print(f"[{time.ctime()}] Generation failed: {e}")
        # Log the full traceback if possible
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error during 3D generation: {str(e)}")


# --- RUN SERVER (Self-Contained) ---

# This block allows the script to be run directly using 'python3 api_app.py'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
