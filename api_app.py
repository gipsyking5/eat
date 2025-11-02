import sys
import os
import json
import base64
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- CRITICAL FIX: Add the Trellis repository root to the Python path ---
# This ensures the worker can find the TrellisImageTo3DPipeline class
sys.path.append(os.path.abspath('/app/trellis_repo')) 

# --- Now try the critical import (Must succeed with the Dockerfile fix) ---
try:
    from TrellisImageTo3DPipeline import TrellisImageTo3DPipeline 
    # Initialize the model pipeline
    pipeline = TrellisImageTo3DPipeline()
    print("TrellisImageTo3DPipeline loaded successfully.")

except ImportError as e:
    print(f"FATAL: Could not import TrellisImageTo3DPipeline. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Failed to initialize TrellisImageTo3DPipeline. Error: {e}")
    sys.exit(1)

# --- FastAPI Setup ---
app = FastAPI()

# Input model for the API request
class TrellisInput(BaseModel):
    image: str # Base64 encoded image
    prompt: str
    
# Function to save base64 image temporarily
def save_base64_image(base64_string: str, filename: str) -> str:
    # Remove header if present (e.g., 'data:image/jpeg;base64,')
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        
    image_data = base64.b64decode(base64_string)
    file_path = f"/tmp/{filename}"
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path

@app.post("/run", name="Generate 3D Model")
async def run_pipeline(input_data: TrellisInput):
    """
    Submits an image and prompt to the Trellis pipeline.
    """
    try:
        timestamp = int(time.time())
        input_image_path = save_base64_image(input_data.image, f"input_{timestamp}.png")
        
        print(f"Running pipeline for prompt: {input_data.prompt} with image: {input_image_path}")
        
        # --- NOTE: Actual pipeline execution goes here ---
        # The line below is where the model runs and is currently mocked until the worker is stable:
        # model_path = pipeline.generate(input_image_path, input_data.prompt) 
        
        # Mocking the successful output path for RunPod's API structure:
        output_dir = f"/workspace/outputs/job_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        model_path = f"{output_dir}/model.glb"
        
        # The RunPod worker returns a list of file objects
        return {
            "status": "COMPLETED",
            "output": [
                # This path is what RunPod will return as the accessible URL to your client.
                {"name": model_path, "mime": "model/gltf-binary"},
            ]
        }

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", name="Health Check")
def health_check():
    """Confirms the FastAPI application is running."""
    return {"status": "RUNNING", "pipeline_ready": True}
