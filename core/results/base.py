from abc import ABC
from datetime import datetime
from typing import Any


class Result(ABC):
    """
    Base class for all results produced by the application.

    A Result represents the outcome of running a model on an image.
    Concrete result types should extend this class with information
    specific to the operation that produced the result.
    """

    def __init__(self, **kwargs):
        self.type:            str            = kwargs.get("type", "unknown")
        self.model_name:      str            = kwargs.get("model_name", "unknown")
        self.created_at:      datetime       = kwargs.get("created_at", datetime.now())
        self.ai_deepfake:     bool           = kwargs["ai_deepfake"]
        self.confidence:      float          = kwargs["confidence"]
        self.image:           str            = kwargs["image"]
        self.returned_obj:    Any            = kwargs["returned_obj"]
        self.metrics:         dict[str, Any] = kwargs.get("metrics", {})
