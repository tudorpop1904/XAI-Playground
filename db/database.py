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

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Ensure storage directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_db()
    print(f"Database initialised at {DB_PATH}")
