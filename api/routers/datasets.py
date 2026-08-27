import kagglehub
from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from data.datasets import DATASETS

router = APIRouter(prefix="/api/v1")

@router.get("/datasets")
def get_datasets():
    return DATASETS

@router.post("/datasets/download")
def download_dataset(slug: str):
    try:
        dpath = kagglehub.dataset_download(slug, force_download=True)
        return {"path": str(dpath)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import shutil
import random
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image

from core.enhancers.super_resolution import SuperResolutionEnhancer

device = "cuda" if torch.cuda.is_available() else "cpu"

@router.post("/datasets/prepare")
def prepare_dataset(slug: str, enhance: bool = False):
    """
    Downloads a Kaggle dataset and creates an index file mapping reals and fakes.
    This bypasses copying files and uses the Kaggle cache directly.
    SuperResolution is ignored for this step to avoid massive processing times.
    """
    try:
        # 1. Download full dataset
        dpath_str = kagglehub.dataset_download(slug, force_download=False)
        dpath = Path(dpath_str)
        
        # 2. Find real/fake images (basic heuristic)
        real_images = list(dpath.rglob("*real*/*.jpg")) + list(dpath.rglob("*real*/*.png"))
        fake_images = list(dpath.rglob("*fake*/*.jpg")) + list(dpath.rglob("*fake*/*.png")) + list(dpath.rglob("*ai*/*.jpg"))
        
        # If the heuristics didn't work nicely, just grab anything
        if not real_images or not fake_images:
            all_imgs = list(dpath.rglob("*.jpg")) + list(dpath.rglob("*.png"))
            half = len(all_imgs) // 2
            real_images = all_imgs[:half]
            fake_images = all_imgs[half:]
            
        # 3. Create output directory for index
        folder_name = slug.split("/")[-1]
        out_dir = Path("storage/datasets")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = out_dir / f"{folder_name}_index.json"
        
        # Save absolute paths to JSON
        import json
        index_data = {
            "real": [str(p.resolve()) for p in real_images],
            "fake": [str(p.resolve()) for p in fake_images]
        }
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)
        
        return {
            "status": "success", 
            "path": str(index_path),
            "reals": len(real_images),
            "fakes": len(fake_images),
            "enhanced": False,
            "cached": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))