import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "app.db")

def get_connection() -> sqlite3.Connection:
    """
    Returns a SQLite database connection with row factory enabled.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialises the SQLite database and creates tables if they don't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()


    # Detection Results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            ai_deepfake BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            metrics TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # XAI Results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xai_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            explainer_method TEXT NOT NULL,
            detection_id INTEGER,
            image_path TEXT NOT NULL,
            heatmap_path TEXT NOT NULL,
            ai_deepfake BOOLEAN NOT NULL,
            confidence REAL NOT NULL,
            metrics TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(detection_id) REFERENCES detection_results(id) ON DELETE SET NULL
        )
    """)

    # XAI Evaluations (The "God" table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xai_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            explainer_method TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            stability_score REAL,
            faithfulness_score REAL,
            sparsity_score REAL
        )
    """)

    # LLM Interpretation Cache
    # Key: (detection_id, xai_result_id, llm_model) — UNIQUE enforces one response per triple.
    # INSERT OR REPLACE allows regenerating a report (overwrites the old text).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_interpretations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id    INTEGER NOT NULL,
            xai_result_id   INTEGER NOT NULL,
            llm_model       TEXT NOT NULL,
            response_text   TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            UNIQUE(detection_id, xai_result_id, llm_model),
            FOREIGN KEY(detection_id)  REFERENCES detection_results(id) ON DELETE CASCADE,
            FOREIGN KEY(xai_result_id) REFERENCES xai_results(id)       ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Ensure storage directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    print(f"Database initialised at {DB_PATH}")
