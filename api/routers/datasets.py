import kagglehub
from fastapi import APIRouter
from fastapi.exceptions import HTTPException

from data.datasets import DATASETS
from api.workers import job_store as _job_store
from api.workers.producer import publish_job

router = APIRouter(prefix="/api/v1")


@router.get("/datasets")
def get_datasets():
    return DATASETS


@router.post("/datasets/download")
def download_dataset(slug: str):
    """
    Asynchronously downloads a Kaggle dataset.
    Returns immediately with a job_id; poll GET /api/v1/jobs/{job_id} for progress.
    """
    try:
        job_id = _job_store.create_job()
        publish_job("kaggle_prepare", {"slug": slug}, job_id)
        return {"job_id": job_id, "status": "queued", "slug": slug}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/prepare")
def prepare_dataset(slug: str, enhance: bool = False):
    """
    Asynchronously downloads and indexes a Kaggle dataset.
    Returns immediately with a job_id; poll GET /api/v1/jobs/{job_id} for progress.
    """
    try:
        job_id = _job_store.create_job()
        publish_job("kaggle_prepare", {"slug": slug, "enhance": enhance}, job_id)
        return {"job_id": job_id, "status": "queued", "slug": slug}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))