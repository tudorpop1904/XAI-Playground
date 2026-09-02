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

    def find_detection(self, model_name: str, filename_or_path: str) -> Optional[Dict[str, Any]]:
        """
        Lookup an existing detection result by model name and filename stem/path.

        Matching strategy (in order of precision):
          1. Exact path match.
          2. LIKE "%<full_stem>%" — stem includes the SHA256 hash suffix, so it's
             unique per image content (e.g. 'real_1_df4116b52ca4').

        We deliberately avoid splitting the stem on '_' to extract a 'base_name'
        (e.g. 'real'), which was causing false positives when multiple images
        share the same prefix word.
        """
        from pathlib import Path
        stem = Path(filename_or_path).stem  # e.g. 'real_1_df4116b52ca4'
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM detection_results
            WHERE model_name = ?
              AND (image_path = ? OR image_path LIKE ?)
            ORDER BY id DESC LIMIT 1
            """,
            (model_name, filename_or_path, f"%{stem}%")
        )
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None

    def find_xai(self, detection_id: int, explainer_method: str) -> Optional[Dict[str, Any]]:
        """
        Lookup an existing XAI result for a specific detection to enable instant cache hits.
        Handles method name aliases (e.g., 'occlusion' vs 'occlusion_sensitivity').
        """
        method_prefix = explainer_method.lower().replace("_sensitivity", "").replace("sensitivity", "")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM xai_results 
            WHERE detection_id = ? 
              AND (explainer_method = ? OR explainer_method LIKE ?)
            ORDER BY id DESC LIMIT 1
            """,
            (detection_id, explainer_method, f"{method_prefix}%")
        )
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None

    def update_xai_evaluation(self, explainer_method: str, dataset_name: str = "Live_Inference") -> None:
        """
        Aggregates running averages for stability, faithfulness, and sparsity
        from all xai_results matching the explainer_method and upserts into xai_evaluations.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT metrics FROM xai_results WHERE explainer_method = ?", (explainer_method,))
        rows = cursor.fetchall()

        stabilities: List[float] = []
        faithfulnesses: List[float] = []
        sparsities: List[float] = []

        for (m_json,) in rows:
            if not m_json:
                continue
            try:
                m = json.loads(m_json) if isinstance(m_json, str) else m_json
                if "stability" in m and m["stability"] is not None:
                    stabilities.append(float(m["stability"]))
                if "faithfulness" in m and m["faithfulness"] is not None:
                    faithfulnesses.append(float(m["faithfulness"]))
                if "sparsity" in m and m["sparsity"] is not None:
                    sparsities.append(float(m["sparsity"]))
            except Exception:
                pass

        avg_stab = round(sum(stabilities) / len(stabilities), 4) if stabilities else None
        avg_faith = round(sum(faithfulnesses) / len(faithfulnesses), 4) if faithfulnesses else None
        avg_spars = round(sum(sparsities) / len(sparsities), 4) if sparsities else None

        cursor.execute(
            "SELECT id FROM xai_evaluations WHERE explainer_method = ? AND dataset_name = ?",
            (explainer_method, dataset_name)
        )
        existing = cursor.fetchone()
        if existing:
            cursor.execute(
                """
                UPDATE xai_evaluations
                SET stability_score = ?, faithfulness_score = ?, sparsity_score = ?
                WHERE id = ?
                """,
                (avg_stab, avg_faith, avg_spars, existing[0])
            )
        else:
            cursor.execute(
                """
                INSERT INTO xai_evaluations
                (explainer_method, dataset_name, stability_score, faithfulness_score, sparsity_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (explainer_method, dataset_name, avg_stab, avg_faith, avg_spars)
            )
        self.conn.commit()

    def sync_all_evaluations(self, dataset_name: str = "Live_Inference") -> None:
        """
        Scans all distinct explainer methods in xai_results and populates
        xai_evaluations with aggregated averages across all historical runs.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT explainer_method FROM xai_results")
        methods = [row[0] for row in cursor.fetchall() if row[0]]
        for m in methods:
            self.update_xai_evaluation(m, dataset_name=dataset_name)

    def get_all_evaluations(self) -> List[Dict[str, Any]]:
        """Returns all aggregated XAI evaluations."""
        # Auto-sync on query if table is currently empty
        cursor = self.conn.cursor()
        cursor.execute("SELECT count(*) FROM xai_evaluations")
        count = cursor.fetchone()[0]
        if count == 0:
            self.sync_all_evaluations()

        cursor.execute("SELECT * FROM xai_evaluations ORDER BY explainer_method ASC")
        return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    #  LLM Interpretation Cache
    # -------------------------------------------------------------------------

    def save_llm_interpretation(
        self,
        detection_id: int,
        xai_result_id: int,
        llm_model: str,
        response_text: str,
    ) -> int:
        """
        Persists a completed LLM forensic report to the database.
        Uses INSERT OR REPLACE so re-running with the same key overwrites the old text.

        Returns the row ID of the inserted/replaced record.
        """
        from datetime import datetime
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO llm_interpretations
                (detection_id, xai_result_id, llm_model, response_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                xai_result_id,
                llm_model,
                response_text,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        row_id = cursor.lastrowid
        logger.info(
            f"[DB] Saved LLM interpretation (detection={detection_id}, xai={xai_result_id}, model={llm_model}) → ID {row_id}"
        )
        return row_id

    def find_llm_interpretation(
        self,
        detection_id: int,
        xai_result_id: int,
        llm_model: str,
    ) -> Optional[str]:
        """
        Returns the cached LLM response text for a given
        (detection_id, xai_result_id, llm_model) triple, or None if not cached.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT response_text FROM llm_interpretations
            WHERE detection_id = ? AND xai_result_id = ? AND llm_model = ?
            LIMIT 1
            """,
            (detection_id, xai_result_id, llm_model),
        )
        row = cursor.fetchone()
        return row[0] if row else None
