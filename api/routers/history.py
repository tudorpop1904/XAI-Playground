from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from db.repositories.result_repository import ResultRepository
from db.database import get_connection

router = APIRouter(prefix="/api/v1")

@router.get("/history")
def get_history(task: str = Query("detection", description="'detection' or 'xai'")) -> list:
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        if task.lower() == "detection":
            return repo.get_all_detections()
        elif task.lower() == "xai":
            return repo.get_all_xai()
        else:
            raise HTTPException(status_code=400, detail=f"Invalid task: {task}. Must be 'detection' or 'xai'")
    finally:
        conn.close()

@router.get("/history/{id}")
def get_history_by_id(id: int, task: Optional[str] = Query(None, description="'detection' or 'xai'")) -> dict:
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        if task and task.lower() == "detection":
            res = repo.get_detection_by_id(id)
        elif task and task.lower() == "xai":
            res = repo.get_xai_by_id(id)
        else:
            res = repo.get_detection_by_id(id) or repo.get_xai_by_id(id)
            
        if not res:
            raise HTTPException(status_code=404, detail=f"Result with id {id} not found")
        return res
    finally:
        conn.close()