"""
core/results/explanation.py
============================
Dataclass representing the result of running an XAI explainer on a
detector's prediction.

An XAIResult captures:
  - The heatmap tensor (which regions mattered for the prediction)
  - Which explainer method produced it
  - Performance metrics (time, RAM, forward passes)
  - A reference to the explainer that produced it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch

from .base import Result

if TYPE_CHECKING:
    from core.detectors.base import AbstractBaseDetector
    from core.explainers.base import BaseExplainer


class XAIResult(Result):
    """
    Result produced by an XAI explainer method.

    In addition to the common Result fields, an XAI result stores:
      - The explanation heatmap (same spatial dimensions as the input)
      - The method that produced it
      - Performance metrics for benchmarking
    """

    index = 1

    def __init__(self, **kwargs):
        kwargs["type"] = "explanation"
        self.explainer = kwargs["explainer"]
        kwargs["model_name"] = f"{self.explainer.method_name}_{XAIResult.index}"
        XAIResult.index += 1
        
        super().__init__(**kwargs)
        
        self.method_used: str = self.explainer.method_name
        
        # Merge any explicitly passed metric kwargs into self.metrics without hardcoded dummy defaults
        for key in ("forward_passes", "cell_scores", "sobol_indices"):
            if key in kwargs and key not in self.metrics:
                self.metrics[key] = kwargs[key]
