import os
import ollama
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas.requests import InterpretationRequest

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


def stream_explanation(prompt: str, model_name: str):
    """Stream tokens from Ollama."""
    stream = _client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
            
@router.post("/interpret")
async def interpret_results(request: InterpretationRequest):
    """
    Generate a natural language explanation of the detection and XAI results via Llama 3.1.
    Streams the response back.
    """
    prompt = build_forensic_prompt(request)
    return StreamingResponse(stream_explanation(prompt, request.llm_model), media_type="text/event-stream")
