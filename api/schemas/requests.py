from pydantic import BaseModel

from typing import Any

class InterpretationRequest(BaseModel):
    detector_name: str
    explainer_name: str
    ai_deepfake: bool
    confidence: float
    metrics: dict[str, Any]
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
