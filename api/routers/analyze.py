"""
# api/routers/analyze.py

This module stands as the API router for the "api/v1/analyze" endpoint.
It handles the logic for:
1. Loading the uploaded image.
2. Detecting deepfakes using the selected model.
3. Explaining the detection using the selected explainer.
4. Computing XAI metrics.
5. Saving the results to the database.

"""

import torch
import io
from datetime import datetime
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form

from api.schemas.responses import AnalysisResponse
from core.detectors.base import AbstractBaseDetector
from core.explainers.base import BaseExplainer
from core.enhancers.super_resolution import SuperResolutionEnhancer
from core.utils.metrics import (
    compute_sparsity,
    compute_faithfulness,
    compute_stability,
)
from db.repositories.result_repository import ResultRepository
from db.database import get_connection
from core.utils.logger import get_logger
import time

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    detector: str = Form(...),
    explainer: str = Form(...),
    enhance: bool = Form(False),
    grid_rows: int = Form(4),
    grid_cols: int = Form(4),
    n_samples: int = Form(64)
):
    from api.main import IMAGES
    
    logger.info(f"Received /analyze request. Detector: {detector}, Explainer: {explainer}, Enhance: {enhance}")
    start_time = time.time()

    # Read uploaded file
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    transform = transforms.ToTensor()
    image_tensor = transform(img).unsqueeze(0)
    
    filename_stem = Path(file.filename).stem
    tensor_path = IMAGES / f"{filename_stem}.pt"
    
    # Check for enhancements
    if enhance:
        image_tensor = SuperResolutionEnhancer().enhance(image_tensor)
        tensor_path = IMAGES / f"{filename_stem}_enhanced.pt"
        
    torch.save(image_tensor, tensor_path)

    # Fetch DetectionResults
    logger.info(f"Running detection using model: {detector}")
    detector_model = AbstractBaseDetector.get_by_name(detector)
    detection = detector_model.predict(str(tensor_path))
    logger.info(f"Detection finished. Is Deepfake: {detection.ai_deepfake} (Confidence: {detection.confidence:.2f})")

    # Fetch XAIResults
    logger.info(f"Running XAI explanation using method: {explainer}")
    kwargs = {}
    if explainer.upper() in ["OCCLUSION", "PMI", "SOBOL"]:
        kwargs["grid_rows"] = grid_rows
        kwargs["grid_cols"] = grid_cols
    if explainer.upper() == "SOBOL":
        kwargs["n_samples"] = n_samples
        
    explainer_model = BaseExplainer.get_explainer(explainer, **kwargs)
    explanation = explainer_model.explain(
        detector_model,
        str(tensor_path),
        int(detection.ai_deepfake)
    )
    logger.info(f"XAI explanation generated successfully.")

    heatmap = explanation.returned_obj

    # Save heatmap to disk for frontend display
    heatmap_path = IMAGES / f"{tensor_path.stem}_{explainer}_heatmap.pt"
    torch.save(heatmap, heatmap_path)

    # Compute XAI metrics
    sparsity = compute_sparsity(heatmap)
    faithfulness = compute_faithfulness(
        detector_model,
        image_tensor,
        heatmap,
        int(detection.ai_deepfake)
    )
    stability = compute_stability(
        explainer_model,
        str(tensor_path),
        int(detection.ai_deepfake),
        detector_model
    )

    # Update the actual domain object with newly computed metrics
    if not hasattr(explanation, "metrics") or explanation.metrics is None:
        explanation.metrics = {}
        
    explanation.metrics.update({
        "sparsity": sparsity,
        "faithfulness": faithfulness,
        "stability": stability
    })

    # Save XAIResult to disk for persistence
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        repo.save_xai(
            explanation,  # Pass the actual XAIResult domain object
            None,         # detection_id is None for now
            str(tensor_path),
            str(heatmap_path)
        )
        logger.info(f"Results successfully persisted to SQLite database.")
    except Exception as e:
        logger.error(f"Failed to persist results to database: {e}")
        raise
    finally:
        conn.close()

    # Create combined metrics for the API response
    combined_metrics = detection.metrics.copy()
    combined_metrics.update(explanation.metrics)

    # Return the Pydantic DTO
    response = AnalysisResponse(
        type=detection.type,
        model_name=detection.model_name,
        ai_deepfake=detection.ai_deepfake,
        confidence=detection.confidence,
        returned_obj=str(heatmap_path),
        metrics=combined_metrics,
        created_at=datetime.now(),
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Request completed in {elapsed_time:.2f} seconds.")

    return response.model_dump()
