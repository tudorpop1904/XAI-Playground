# pyrefly: ignore [missing-import]
import shutil

# pyrefly: ignore [missing-import]
import subprocess

# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException

from api.workers.train_worker import train
from core.utils.paths import BASE_DIR, MODELS_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
IS_SLURM = shutil.which("sbatch") is not None

router = APIRouter(prefix="/api/v1")

from api.schemas.requests import TrainRequest


@router.post("/models/train")
def train_model(req: TrainRequest):
    """
    Train a model on a pre-prepared dataset subset with custom forensic channels.
    """

    if (IS_SLURM):
        cmd = [
            "sbatch",
            "--parsable",
            f"--chdir={BASE_DIR}",
            str(BASE_DIR / "scripts" / "train.slurm"),
            "--payload",
            req.model_dump_json(),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"SBATCH job submission failed: {e.stderr.strip()}")

        slurm_output = result.stdout.strip()
        slurm_job_id = slurm_output.split(";")[0]

        return {
            "status": "queued",
            "slurm_job_id": slurm_job_id,
            "message": f"Training job dispatched to compute1 (GPU). Job ID: {slurm_job_id}."
        }

    try:
        return train(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.get("/models/train/{slurm_job_id}")
def train_status(slurm_job_id: str):
    """
    Check the status of a remote training job.

    Args:
        slurm_job_id (str): The ID of the job to check.

    Returns:
        dict: The status of the job.
    """
    if not slurm_job_id.isdigit():
        raise HTTPException(status_code=400, detail="Invalid job ID")
    
    out = subprocess.run(
        ["sacct", "-j", slurm_job_id, "-X", "--noheader", "-o", "State"], 
        capture_output=True, 
        text=True,
    ).stdout.strip()

    state = out.split()[0] if out else "UNKNOWN"

    return {"slurm_job_id": slurm_job_id, "state": state}


@router.get("/models")
def list_models():
    """
    Returns a list of trained models available in storage, along with their metrics.
    """
    import json
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    models = []
    for pth_file in MODELS_DIR.glob("*.pth"):
        model_name = pth_file.stem
        
        # Try to load history
        history = {}
        json_file = MODELS_DIR / f"{model_name}.json"
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    history = json.load(f)
            except Exception:
                pass
                
        models.append({
            "name": model_name,
            "history": history
        })
        
    return models
