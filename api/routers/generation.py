import io
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from pathlib import Path
import time
from core.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")

class GenerateRequest(BaseModel):
    prompt: str
    mode: str = "cloud" # "cloud" or "local"
    hf_token: str = ""
    model_id: str = "runwayml/stable-diffusion-v1-5"

# Global cache for the local pipeline
_local_pipeline = None

def get_local_pipeline(model_id: str):
    global _local_pipeline
    import torch
    from diffusers import StableDiffusionPipeline
    
    if _local_pipeline is None:
        logger.info(f"Loading local Stable Diffusion model: {model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            _local_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                requires_safety_checker=False,
                safety_checker=None
            )
            _local_pipeline = _local_pipeline.to(device)
            logger.info("Local model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")
            
    return _local_pipeline

@router.post("/generate")
def generate_image(req: GenerateRequest):
    """
    Generate a synthetic Deepfake image using Hugging Face Diffusion Models.
    Supports either 'cloud' (HF Inference API) or 'local' (diffusers on GPU).
    """
    start_time = time.time()
    
    if req.mode == "cloud":
        if not req.hf_token:
            raise HTTPException(status_code=400, detail="Hugging Face API token is required for cloud mode.")
            
        api_url = f"https://api-inference.huggingface.co/models/{req.model_id}"
        headers = {"Authorization": f"Bearer {req.hf_token}"}
        
        try:
            logger.info(f"Requesting HF API for prompt: {req.prompt}")
            response = requests.post(api_url, headers=headers, json={"inputs": req.prompt})
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cloud generation failed: {e}")
            
    elif req.mode == "local":
        try:
            pipe = get_local_pipeline(req.model_id)
            logger.info(f"Generating locally for prompt: {req.prompt}")
            image = pipe(req.prompt).images[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local generation failed: {e}")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'cloud' or 'local'.")
        
    elapsed_time = time.time() - start_time
    logger.info(f"Generation completed in {elapsed_time:.2f} seconds.")
    
    # Save the generated image
    from api.main import IMAGES
    IMAGES.mkdir(parents=True, exist_ok=True)
    save_path = IMAGES / f"generated_{int(time.time())}.jpg"
    image.save(save_path)
    
    # Return path relative to project root or absolute, but we need it available for frontend
    return {
        "status": "success",
        "image_path": str(save_path),
        "time_taken": elapsed_time
    }
