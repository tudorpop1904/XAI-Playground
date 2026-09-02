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

import hashlib
from typing import Literal

router = APIRouter(prefix="/api/v1")

@router.get("/evaluations")
async def get_evaluations():
    """Returns aggregated live XAI benchmark metrics grouped by explainer method."""
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        return repo.get_all_evaluations()
    finally:
        conn.close()

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    detector: str = Form(..., min_length=1, description="Detector model name"),
    explainer: Literal["grad_cam", "vanilla_saliency", "occlusion", "pmi", "sobol"] = Form(..., description="XAI method"),
    enhance: bool = Form(False, description="Apply super-resolution before inference"),
    grid_rows: int = Form(4, ge=2, le=16, description="Grid rows for perturbation methods (2-16)"),
    grid_cols: int = Form(4, ge=2, le=16, description="Grid columns for perturbation methods (2-16)"),
    n_samples: int = Form(64, ge=16, le=256, description="Number of Monte Carlo masks for Sobol (16-256)")
):
    from api.main import IMAGES
    
    logger.info(f"Received /analyze request. Detector: {detector}, Explainer: {explainer}, Enhance: {enhance}")
    start_time = time.time()

    # Read uploaded file and compute content hash
    image_bytes = await file.read()
    file_hash = hashlib.sha256(image_bytes).hexdigest()[:12]
    filename_stem = f"{Path(file.filename).stem}_{file_hash}"
    
    tensor_path = IMAGES / f"{filename_stem}.pt"
    if enhance:
        tensor_path = IMAGES / f"{filename_stem}_enhanced.pt"

    # Save tensor to disk if not already cached
    if not tensor_path.exists():
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.ToTensor()
        image_tensor = transform(img).unsqueeze(0)
        if enhance:
            image_tensor = SuperResolutionEnhancer().enhance(image_tensor)
        torch.save(image_tensor, tensor_path)
    else:
        image_tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)

    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        
        # 1. SMART CACHE STEP 1: Detection Result Reuse
        existing_det = repo.find_detection(detector, str(tensor_path))
        if not existing_det:
            existing_det = repo.find_detection(detector, file.filename)

        if existing_det:
            detection_id = existing_det["id"]
            ai_deepfake = bool(existing_det["ai_deepfake"])
            confidence = float(existing_det["confidence"])
            det_metrics = existing_det.get("metrics") or {}
            logger.info(f"[CACHE HIT] Reusing existing detection (ID: {detection_id}) for {detector}.")
            detector_model = AbstractBaseDetector.get_by_name(detector)
        else:
            logger.info(f"Running detection using model: {detector}")
            detector_model = AbstractBaseDetector.get_by_name(detector)
            detection = detector_model.predict(str(tensor_path))
            ai_deepfake = bool(detection.ai_deepfake)
            confidence = float(detection.confidence)
            det_metrics = detection.metrics.copy() if detection.metrics else {}
            detection_id = repo.save_detection(detection, str(tensor_path))
            logger.info(f"Detection saved to DB (ID: {detection_id}). Deepfake: {ai_deepfake} ({confidence:.2f})")

        # 2. SMART CACHE STEP 2: XAI Heatmap & Metrics Reuse
        existing_xai = repo.find_xai(detection_id, explainer)
        cached_heatmap_path = None
        if existing_xai and existing_xai.get("heatmap_path"):
            hp = Path(existing_xai["heatmap_path"])
            if hp.exists():
                cached_heatmap_path = hp
            elif (IMAGES / hp.name).exists():
                cached_heatmap_path = IMAGES / hp.name
            else:
                fallback = IMAGES / f"{tensor_path.stem}_{explainer}_heatmap.pt"
                if fallback.exists():
                    cached_heatmap_path = fallback

        if existing_xai and cached_heatmap_path is not None:
            logger.info(f"[CACHE HIT] Reusing existing XAI heatmap (ID: {existing_xai['id']}, Path: {cached_heatmap_path}) for {explainer}.")
            heatmap_path = cached_heatmap_path
            xai_metrics = existing_xai.get("metrics") or {}
            xai_result_id = existing_xai["id"]
        else:
            logger.info(f"Computing new XAI explanation using: {explainer}")
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
                int(ai_deepfake)
            )

            heatmap = explanation.returned_obj
            heatmap_path = IMAGES / f"{tensor_path.stem}_{explainer}_heatmap.pt"
            torch.save(heatmap, heatmap_path)

            sparsity = compute_sparsity(heatmap)
            faithfulness = compute_faithfulness(
                detector_model,
                image_tensor,
                heatmap,
                int(ai_deepfake)
            )
            stability = compute_stability(
                explainer_model,
                str(tensor_path),
                int(ai_deepfake),
                detector_model
            )

            if not hasattr(explanation, "metrics") or explanation.metrics is None:
                explanation.metrics = {}

            explanation.metrics.update({
                "sparsity": sparsity,
                "faithfulness": faithfulness,
                "stability": stability
            })
            xai_metrics = explanation.metrics

            xai_result_id = repo.save_xai(
                explanation,
                detection_id,
                str(tensor_path),
                str(heatmap_path)
            )
            # Update live running averages in xai_evaluations table
            repo.update_xai_evaluation(explainer)
            logger.info(f"Saved new XAI result and updated live evaluations for {explainer}.")

    finally:
        conn.close()

    # Combine metrics
    combined_metrics = det_metrics.copy()
    combined_metrics.update(xai_metrics)

    elapsed_time = time.time() - start_time
    logger.info(f"Request completed in {elapsed_time:.2f} seconds.")

    response = AnalysisResponse(
        type="DETECTION",
        model_name=detector,
        ai_deepfake=ai_deepfake,
        confidence=confidence,
        returned_obj=str(heatmap_path),
        metrics=combined_metrics,
        created_at=datetime.now(),
        detection_id=detection_id,
        xai_result_id=xai_result_id,
    )

    return response.model_dump()
