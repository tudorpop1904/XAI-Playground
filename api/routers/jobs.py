"""
api/routers/jobs.py

Provides the GET /api/v1/jobs/{job_id} polling endpoint.
UI clients call this endpoint repeatedly (every 2s) to check the status
of an async job submitted to the RabbitMQ queue.
"""

from fastapi import APIRouter, HTTPException
from api.workers import job_store

router = APIRouter(prefix="/api/v1")


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Returns the current status of an async background job.

    Status values:
      - "queued"  : Job published to RabbitMQ, not yet picked up.
      - "running" : Worker is currently executing the job.
      - "done"    : Job completed successfully. 'result' contains the output.
      - "error"   : Job failed. 'error' contains the error message.
    """
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
