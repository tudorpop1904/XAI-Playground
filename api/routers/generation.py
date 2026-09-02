import io
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from pathlib import Path
import time
from core.utils.logger import get_logger
from api.workers import job_store as _job_store
from api.workers.producer import publish_job

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")

from api.schemas.requests import GenerateRequest

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

    - mode="cloud"  : Async via RabbitMQ — returns {job_id, status="queued"} immediately.
                      Poll GET /api/v1/jobs/{job_id} for result.
    - mode="local"  : Synchronous — runs diffusers pipeline on local GPU. Returns image_path directly.
    """
    if req.mode == "cloud":
        if not req.hf_token:
            raise HTTPException(status_code=400, detail="Hugging Face API token is required for cloud mode.")

        try:
            job_id = _job_store.create_job()
            publish_job(
                "hf_generate",
                {
                    "prompt": req.prompt,
                    "model_id": req.model_id,
                    "hf_token": req.hf_token,
                },
                job_id,
            )
            logger.info(f"Queued HF cloud generation job {job_id} for model {req.model_id}")
            return {"job_id": job_id, "status": "queued", "model_id": req.model_id}
        except Exception as e:
            logger.error(f"Failed to queue HF generation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to queue generation job: {e}")

    elif req.mode == "local":
        # Local GPU pipeline — stays synchronous (no external network calls)
        try:
            start_time = time.time()
            pipe = get_local_pipeline(req.model_id)
            logger.info(f"Generating locally for prompt: {req.prompt}")
            image = pipe(req.prompt).images[0]
            elapsed_time = time.time() - start_time

            from api.main import IMAGES
            IMAGES.mkdir(parents=True, exist_ok=True)
            save_path = IMAGES / f"generated_{int(time.time())}.jpg"
            image.save(save_path)

            logger.info(f"Local generation completed in {elapsed_time:.2f}s")
            return {
                "status": "success",
                "image_path": str(save_path),
                "time_taken": elapsed_time,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local generation failed: {e}")
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'cloud' or 'local'.")
