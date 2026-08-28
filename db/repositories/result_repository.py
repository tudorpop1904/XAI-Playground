import json
import sqlite3
from typing import Optional, Any, Dict, List

from core.results.detection import DetectionResult
from core.results.explanation import XAIResult
from core.utils.logger import get_logger

logger = get_logger(__name__)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Helper to convert a sqlite3.Row to a dictionary and parse JSON fields."""
    d = dict(row)
    if "ai_deepfake" in d and d["ai_deepfake"] is not None:
        d["ai_deepfake"] = bool(d["ai_deepfake"])
    if "probabilities" in d and isinstance(d["probabilities"], str):
        try:
            d["probabilities"] = json.loads(d["probabilities"])
        except Exception:
            pass
    if "metrics" in d and isinstance(d["metrics"], str):
        try:
            d["metrics"] = json.loads(d["metrics"])
        except Exception:
            pass
    return d


class ResultRepository:
    """
    Repository class to persist Result objects to the SQLite database.
    This separates domain logic (Results) from persistence logic (SQLite).
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save_detection(self, result: DetectionResult, image_path: Optional[str] = None) -> int:
        """
        Saves a DetectionResult and returns its database ID.

        Schema (detection_results):
            id, model_name, image_path, ai_deepfake, confidence,
            probabilities, metrics, created_at
        """
        img_path = image_path or getattr(result, "image", "")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO detection_results
            (model_name, image_path, ai_deepfake, confidence, probabilities, metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.model_name,
                img_path,
                int(result.ai_deepfake),
                result.confidence,
                json.dumps(result.returned_obj),   # probabilities tensor/dict
                json.dumps(result.metrics),         # timing/eval metrics dict
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
        image_path: Optional[str] = None,
        heatmap_path: Optional[str] = None
    ) -> int:
        """
        Saves an XAIResult and returns its database ID.

        Schema (xai_results):
            id, explainer_method, detection_id, image_path, heatmap_path,
            ai_deepfake, confidence, metrics, created_at
        """
        img_path = image_path or getattr(result, "image", "")
        heat_path = heatmap_path or ""
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
                img_path,
                heat_path,
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

    def get_all_detections(self) -> List[Dict[str, Any]]:
        """Returns all detection results as clean dictionaries."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM detection_results ORDER BY created_at DESC")
        return [_row_to_dict(row) for row in cursor.fetchall()]

    def get_all_xai(self) -> List[Dict[str, Any]]:
        """Returns all XAI results as clean dictionaries."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM xai_results ORDER BY created_at DESC")
        return [_row_to_dict(row) for row in cursor.fetchall()]

    def get_detection_by_id(self, detection_id: int) -> Optional[Dict[str, Any]]:
        """Returns a single detection result by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM detection_results WHERE id = ?", (detection_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None

    def get_xai_by_id(self, xai_id: int) -> Optional[Dict[str, Any]]:
        """Returns a single XAI result by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM xai_results WHERE id = ?", (xai_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None

    def get_xai_for_detection(self, detection_id: int) -> List[Dict[str, Any]]:
        """Returns all XAI results linked to a given detection run."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM xai_results WHERE detection_id = ? ORDER BY created_at DESC",
            (detection_id,)
        )
        return [_row_to_dict(row) for row in cursor.fetchall()]


