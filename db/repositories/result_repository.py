import json
import sqlite3
from typing import Optional

from core.results.detection import DetectionResult
from core.results.explanation import XAIResult
from core.utils.logger import get_logger

logger = get_logger(__name__)

class ResultRepository:
    """
    Repository class to persist Result objects to the SQLite database.
    This separates domain logic (Results) from persistence logic (SQLite).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save_detection(self, result: DetectionResult, image_path: str) -> int:
        """
        Saves a DetectionResult and returns its database ID.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO detection_results 
            (model_name, image_path, ai_deepfake, confidence, probabilities, elapsed_sec, peak_ram_mb, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.model_name,
                image_path,
                result.ai_deepfake,
                result.confidence,
                json.dumps(result.returned_obj),
                result.elapsed_seconds,
                result.peak_ram_mb,
                result.created_at.isoformat()
            )
        )
        self.conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"[DB] Saved DetectionResult {result.model_name} to DB (ID: {last_id})")
        return last_id

    def save_xai(
        self, 
        result: XAIResult, 
        detection_id: Optional[int], 
        image_path: str, 
        heatmap_path: str
    ) -> int:
        """
        Saves an XAIResult and returns its database ID.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO xai_results 
            (explainer_method, detection_id, image_path, heatmap_path, ai_deepfake, confidence, metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.method_used,
                detection_id,
                image_path,
                heatmap_path,
                result.ai_deepfake,
                result.confidence,
                json.dumps(result.metrics),
                result.created_at.isoformat()
            )
        )
        self.conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"[DB] Saved XAIResult {result.method_used} to DB (ID: {last_id})")
        return last_id
    
    def get_all_detections(self) -> list[DetectionResult]:
        """
        Returns all detection results from the SQLite database.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM detection_results
            """
        )
        return cursor.fetchall()

    def get_all_xai(self) -> list[XAIResult]:
        """
        Returns all XAI results from the SQLite database.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM xai_results
            """
        )
        return cursor.fetchall()

    def get_by_id(self, id: int) -> dict:
        """
        Returns a single result by its ID.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM results WHERE id = ?
            """,
            (id,)
        )
        return cursor.fetchone()

    def get_all(self) -> list[dict]:
        """
        Returns all results from the SQLite database.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM results
            """
        )
        return cursor.fetchall()
