from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Result



class DetectionResult(Result):
    """
    Result produced by a detector.

    In addition to the common Result fields, a detection result stores:
      - The model that produced it
      - The unnormalized model outputs
      - The computed probability for each class, useful for displaying
    class probabilities.
    """

    index = 1

    def __init__(self, **kwargs):
        kwargs["type"] = "detection"
        self.model = kwargs["model"]
        kwargs["model_name"] = f"{self.model.name}_{DetectionResult.index}"
        DetectionResult.index += 1
        
        super().__init__(**kwargs)