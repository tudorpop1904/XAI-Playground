from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "storage" / "models"
DB_DIR = BASE_DIR / "storage"
DATASETS_DIR = BASE_DIR / "storage" / "datasets"
IMAGES_DIR = BASE_DIR / "storage" / "images"