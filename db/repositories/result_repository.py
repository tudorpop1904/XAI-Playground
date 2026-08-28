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

        Schema (detection_results):
            id, model_name, image_path, ai_deepfake, confidence,
            probabilities, metrics, created_at
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO detection_results
            (model_name, image_path, ai_deepfake, confidence, probabilities, metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.model_name,
                image_path,
                int(result.ai_deepfake),
                result.confidence,
                json.dumps(result.returned_obj),   # probabilities tensor/list
                json.dumps(result.metrics),         # timing/RAM/eval metrics dict
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

        Schema (xai_results):
            id, explainer_method, detection_id, image_path, heatmap_path,
            ai_deepfake, confidence, metrics, created_at
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
                int(result.ai_deepfake),
                result.confidence,
                json.dumps(result.metrics),
                result.created_at.isoformat()
            )
        )
        self.conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"[DB] Saved XAIResult {result.method_used} to DB (ID: {last_id})")
        return last_id

    def get_all_detections(self) -> list:
        """Returns all detection results as raw Row objects."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM detection_results ORDER BY created_at DESC")
        return cursor.fetchall()

    def get_all_xai(self) -> list:
        """Returns all XAI results as raw Row objects."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM xai_results ORDER BY created_at DESC")
        return cursor.fetchall()

    def get_xai_for_detection(self, detection_id: int) -> list:
        """Returns all XAI results linked to a given detection run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM xai_results WHERE detection_id = ? ORDER BY created_at DESC",
            (detection_id,)
        )
        return cursor.fetchall()

