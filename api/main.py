from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import os

from db.database import init_db
from api.routers import analyze, enhance, history, interpret, datasets, training, generation

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure storage directories exist before anything else
    storage = Path(__file__).parent.parent / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "images").mkdir(parents=True, exist_ok=True)
    (storage / "models").mkdir(parents=True, exist_ok=True)
    (storage / "datasets").mkdir(parents=True, exist_ok=True)
    # Initialise SQLite schema (CREATE TABLE IF NOT EXISTS — idempotent)
    init_db()
    yield

app = FastAPI(title="Playground", description="Playground API", version="1.0.0", lifespan=lifespan)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Allow CORS from both local dev and Docker environments
_default_origins = "http://localhost:8501,http://ui:8501"
origins = os.environ.get("CORS_ORIGINS", _default_origins).split(",")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# storage/images is created by lifespan above; just hold the reference here
IMAGES = Path(__file__).parent.parent.joinpath("storage").joinpath("images")

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
