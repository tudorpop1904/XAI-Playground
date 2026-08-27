from fastapi import APIRouter
from db.repositories.result_repository import ResultRepository
from db.database import get_connection

router = APIRouter(prefix="/api/v1")

@router.get("/history")
def get_history(task: str) -> list:
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        if task == "detection":
            return repo.get_all_detections()
        elif task == "xai":
            return repo.get_all_xai()
        else:
            raise ValueError(f"Invalid task: {task}")
    finally:
        conn.close()

@router.get("/history/{id}")
def get_history_by_id(id: int) -> dict:
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        return repo.get_by_id(id)
    finally:
        conn.close()