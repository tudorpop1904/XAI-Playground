from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import os

from api.routers import analyze, enhance, history, interpret, datasets, training, generation

app = FastAPI(title="Playground", description="Playground API", version="1.0.0")
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

# storage/images directory is one level up from api/ (i.e. playground-app/storage/images)
IMAGES = Path(__file__).parent.parent.joinpath("storage").joinpath("images")

# Ensure the directory exists
IMAGES.mkdir(parents=True, exist_ok=True)

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
