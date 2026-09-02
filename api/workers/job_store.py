"""
api/workers/job_store.py

Thread-safe in-memory store for async job results.
Maps job_id -> {status, result, error, created_at}

Statuses:
  - "queued"    : published to RabbitMQ, awaiting pickup
  - "running"   : worker started processing
  - "done"      : completed successfully; result populated
  - "error"     : failed; error populated
"""

import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

_lock = threading.Lock()
_store: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    """Creates a new job entry and returns its unique job_id."""
    job_id = str(uuid.uuid4())
    with _lock:
        _store[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
        }
    return job_id


def update_job(
    job_id: str,
    status: str,
    result: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    """Updates the status (and optionally result/error) for a given job."""
    with _lock:
        if job_id not in _store:
            return
        _store[job_id]["status"] = status
        if result is not None:
            _store[job_id]["result"] = result
        if error is not None:
            _store[job_id]["error"] = error


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Returns the job dict for a given job_id, or None if not found."""
    with _lock:
        return _store.get(job_id)
