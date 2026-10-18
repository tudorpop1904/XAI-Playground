from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import os

from db.database import init_db
from api.routers import analyze, enhance, history, interpret, datasets, training, generation, jobs

# Ensure storage directories exist at module load time (StaticFiles requires directory to exist)
STORAGE = Path(__file__).parent.parent / "storage"
IMAGES = STORAGE / "images"
MODELS = STORAGE / "models"
DATASETS = STORAGE / "datasets"

for d in (STORAGE, IMAGES, MODELS, DATASETS):
    d.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage directories exist before anything else
    for d in (STORAGE, IMAGES, MODELS, DATASETS):
        d.mkdir(parents=True, exist_ok=True)
    # Initialise SQLite schema (CREATE TABLE IF NOT EXISTS — idempotent)
    init_db()
    # Start RabbitMQ consumer background thread if enabled
    if os.environ.get("RABBITMQ_ENABLED", "true").lower() in ("1", "true", "yes"):
        try:
            from api.workers.consumer import start_consumer_thread
            start_consumer_thread()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not start RabbitMQ consumer: {e}")
    yield

app = FastAPI(title="Playground", description="Playground API", version="1.0.0", lifespan=lifespan)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Allow CORS from both local dev and Docker environments
_default_origins = "http://localhost:8501,http://localhost:8502,http://localhost:8503,http://ui:8501"
origins = os.environ.get("CORS_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve heatmap/uploaded images
app.mount("/storage/images", StaticFiles(directory=str(IMAGES)), name="images")

# Register all routers
app.include_router(analyze.router)
app.include_router(enhance.router)
app.include_router(history.router)
app.include_router(interpret.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(generation.router)
app.include_router(jobs.router)
