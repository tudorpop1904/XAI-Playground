"""
api/routers/interpret.py

LLM-based forensic interpretation of detection + XAI results via Ollama.

Cache behaviour
---------------
If the request includes both `detection_id` and `xai_result_id`, the endpoint:
  1. Checks `llm_interpretations` in SQLite for a cached response.
  2. CACHE HIT  → streams the stored text instantly (fake-stream, no Ollama call).
  3. CACHE MISS → generates with Ollama, accumulates the full text, saves to DB,
                  then streams it to the client.

This eliminates redundant Ollama calls for identical (image, detector, explainer, model) tuples.
"""

import os
import ollama
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from typing import Optional

from api.schemas.requests import InterpretationRequest
from db.database import get_connection
from db.repositories.result_repository import ResultRepository
from core.utils.logger import get_logger

logger = get_logger(__name__)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_client = ollama.Client(host=OLLAMA_HOST)
router = APIRouter(prefix="/api/v1")


@router.get("/interpret/status")
def check_ollama_available() -> dict:
    """Check if the local Ollama daemon is reachable and list models."""
    try:
        models = _client.list()
        names = [m.model for m in models.models]

        if names:
            return {"status": True, "models": names}
        return {"status": False, "message": "Ollama is running but no models found. Run `ollama pull llama3.1`."}

    except Exception as e:
        return {"status": False, "message": f"Cannot reach Ollama at {OLLAMA_HOST}: {e}"}


@router.get("/interpret/cached")
def get_cached_interpretation(
    detection_id: int = Query(...),
    xai_result_id: int = Query(...),
    llm_model: str = Query(OLLAMA_MODEL),
) -> dict:
    """
    Checks whether a cached LLM interpretation exists for the given triple.

    Returns:
        {"cached": true,  "response_text": "..."} if found.
        {"cached": false}                          if not found.
    """
    conn = get_connection()
    try:
        repo = ResultRepository(conn)
        cached = repo.find_llm_interpretation(detection_id, xai_result_id, llm_model)
        if cached:
            return {"cached": True, "response_text": cached}
        return {"cached": False}
    finally:
        conn.close()


def build_forensic_prompt(request: InterpretationRequest, deployment_env: str = "local") -> str:
    """Build prompt comparing XAI methods for AI-generated image detection."""
    detection_label = "Deepfake" if request.ai_deepfake else "Real"
    
    parts = [
        "You are a digital forensics analyst explaining AI-generated image detection "
        "to a researcher comparing Explainable AI methods.\n",
        f"Detection: **{detection_label}** (confidence {request.confidence:.1%})\n",
        f"Detector Model Used: {request.detector_name}\n",
        f"Explainer Model Used: {request.explainer_name}\n",
        f"Deployment environment: {deployment_env}\n",
        "\nThe following visual XAI metrics were extracted for this explanation heatmap:\n",
    ]

    parts.append(
        f"- Sparsity: {request.metrics.get('sparsity', 'N/A')} (Higher is more concise)\n"
        f"- Faithfulness: {request.metrics.get('faithfulness', 'N/A')} (Drop in confidence when highlighted pixels are removed. Higher is better)\n"
        f"- Stability: {request.metrics.get('stability', 'N/A')} (Resistance to noise. Higher is better)\n"
        f"- Forward passes required: {request.metrics.get('forward_passes', 'N/A')}\n"
    )
    
    parts.append(
        "\nWrite a concise report (3-5 paragraphs) covering:\n"
        "1. What the detection result means for this image.\n"
        "2. An evaluation of the chosen XAI Explainer based on the provided metrics (sparsity, faithfulness, stability).\n"
        "3. A practical recommendation on whether this specific Explainer balances fidelity, stability, and cost well for this image.\n"
        "Use clear academic Romanian-friendly English (bilingual terms OK). "
        "Do not invent numeric heatmap values — refer only to the metrics provided.\n"
    )
    return "".join(parts)


def _fake_stream(text: str):
    """Yields a cached response as a stream (word-by-word) to preserve UX."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")


@router.post("/interpret")
async def interpret_results(request: InterpretationRequest):
    """
    Generate a natural language forensic explanation via Llama 3.1 (Ollama).

    - If detection_id + xai_result_id are provided and a cached response exists
      in SQLite → streams the cached text instantly (no Ollama call).
    - Otherwise → generates with Ollama, saves result to DB, streams to client.
    """
    can_cache = (
        request.detection_id is not None
        and request.xai_result_id is not None
    )

    if can_cache:
        conn = get_connection()
        try:
            repo = ResultRepository(conn)
            cached_text = repo.find_llm_interpretation(
                request.detection_id,
                request.xai_result_id,
                request.llm_model,
            )
        finally:
            conn.close()

        if cached_text:
            logger.info(
                f"[LLM CACHE HIT] detection={request.detection_id}, "
                f"xai={request.xai_result_id}, model={request.llm_model}"
            )
            return StreamingResponse(
                _fake_stream(cached_text),
                media_type="text/event-stream",
                headers={"X-Cache": "HIT"},
            )

    # — CACHE MISS or no IDs provided: generate with Ollama —
    prompt = build_forensic_prompt(request)
    logger.info(
        f"[LLM CACHE MISS] Generating with Ollama model={request.llm_model}"
        + (f" (detection={request.detection_id}, xai={request.xai_result_id})" if can_cache else "")
    )

    if can_cache:
        # Accumulate full response, save to DB, then stream
        def generate_and_save():
            accumulated = []
            stream = _client.chat(
                model=request.llm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    accumulated.append(content)
                    yield content

            full_text = "".join(accumulated)
            if full_text:
                conn = get_connection()
                try:
                    ResultRepository(conn).save_llm_interpretation(
                        request.detection_id,
                        request.xai_result_id,
                        request.llm_model,
                        full_text,
                    )
                finally:
                    conn.close()

        return StreamingResponse(
            generate_and_save(),
            media_type="text/event-stream",
            headers={"X-Cache": "MISS"},
        )
    else:
        # No IDs — plain streaming without caching
        def stream_plain():
            stream = _client.chat(
                model=request.llm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

        return StreamingResponse(stream_plain(), media_type="text/event-stream")
