from datetime import datetime
from typing import Dict, Any, Optional
from core.results.base import Result
from pydantic import BaseModel

class AnalysisResponse(BaseModel):
    type: str = "unknown"
    model_name: str = "unknown"
    ai_deepfake: bool = False
    confidence: float = 0.0
    returned_obj: Optional[Any] = None
    metrics: Dict[str, Any] = {}
    created_at: Optional[datetime] = None
    image: Optional[str] = None
    # DB IDs for LLM cache lookups
    detection_id: Optional[int] = None
    xai_result_id: Optional[int] = None
    
    def to_result(self) -> Result:
        return Result(
            type=self.type,
            model_name=self.model_name,
            ai_deepfake=self.ai_deepfake,
            confidence=self.confidence,
            returned_obj=self.returned_obj,
            created_at=self.created_at,
            image=self.image,
            metrics=self.metrics,
        )
