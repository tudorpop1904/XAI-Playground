"""
core/xai/explainers/base.py
============================
Abstract base class for all XAI (Explainable AI) methods.

ARCHITECTURE ROLE
------------------
Explainers are the SERVICE layer for interpretability. They sit between
the detector (model) and the result (output), implementing the Strategy
pattern:

    ┌──────────────┐     ┌────────────────┐     ┌────────────┐
    │   Detector   │────▶│   Explainer    │────▶│  XAIResult  │
    │ (nn.Module)  │     │  (Service)     │     │ (Dataclass) │
    │              │◀────│               │     │             │
    │ get_target   │     │ explain()      │     │ heatmap +   │
    │ _layer()     │     │               │     │ metrics     │
    └──────────────┘     └────────────────┘     └────────────┘

Each concrete explainer implements ONE XAI method (Single Responsibility):
  - GradCAMExplainer                 → gradient-based, uses hooks on target layer
  - VanillaSaliencyExplainer         → gradient-based, uses input gradients
  - OcclusionExplainer               → perturbation-based, occludes grid cells
  - PMIExplainer                     → perturbation-based, reveals grid cells
  - SobolExplainer                   → perturbation-based, Monte Carlo variance

STRATEGY PATTERN
-----------------
All explainers share the same interface: explain(detector, image, class_idx).
This lets the API layer swap methods at runtime without changing any code:

    explainer = EXPLAINER_REGISTRY[method_name]()
    result = explainer.explain(detector, image, target_class)

SEPARATION OF CONCERNS
-----------------------
  - Detectors know HOW to classify (forward pass, predict).
  - Explainers know HOW to explain (hooks, perturbation, gradients).
  - Results know HOW to store/serialise (save_to_database, to_frontend).
  - The API layer orchestrates all three.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from core.utils.paths import MODELS_DIR
from core.utils.logger import get_logger

import torch

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.detectors.base import AbstractBaseDetector
    from core.results.explanation import XAIResult

class BaseExplainer(ABC):
    """
    Abstract base class for all XAI explainer methods.

    Every concrete explainer must implement:
      - explain()       → produce an XAIResult (heatmap + metadata)
      - method_name     → a class-level string identifying the method

    Explainers are STATELESS services — they don't hold model weights
    or training data. They receive a detector and an image, and return
    a result. This makes them easy to test, swap, and compose.
    """

    # Each subclass sets this to a unique identifier, e.g. "grad_cam"
    method_name: str = ""

    @abstractmethod
    def explain(
        self,
        detector: AbstractBaseDetector,
        image_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult:
        """
        Generate an explanation heatmap for the detector's prediction.

        Parameters
        ----------
        detector : AbstractBaseDetector
            The trained detector model to explain. The explainer may
            call detector.forward(), register hooks on its layers, or
            access detector.get_target_layer() — depending on the method.

        image_tensor : torch.Tensor
            Single image tensor of shape [C, H, W], values in [0, 1].

        target_class : int
            The class index to explain (0 = Real, 1 = AI-generated).
            The heatmap shows which regions contributed to THIS class.

        Returns
        -------
        XAIResult
            Contains the heatmap tensor, method metadata, and
            performance metrics.
        """
        pass

    @staticmethod
    def get_explainer(explainer_type: str, **kwargs) -> BaseExplainer:
        """
        Factory method to instantiate an explainer from a string identifier.

        Parameters
        ----------
        explainer_type : str
            The type of explainer to instantiate.
        kwargs : dict
            Additional arguments to pass to the explainer constructor.

        Returns
        -------
        BaseExplainer
            The instantiated explainer.

        Raises
        ------
        ValueError
            If the explainer type is not found.
        """
        from core.explainers.gradient import GradCAMExplainer, VanillaSaliencyExplainer
        from core.explainers.perturbation import OcclusionExplainer, PMIExplainer, SobolExplainer

        EXPLAINERS = {
            "GRAD_CAM": GradCAMExplainer,
            "VANILLA_SALIENCY": VanillaSaliencyExplainer,
            "OCCLUSION": OcclusionExplainer,
            "PMI": PMIExplainer,
            "SOBOL": SobolExplainer
        }

        explainer = EXPLAINERS.get(explainer_type.upper())

        if explainer is None:
            raise ValueError("Explainer type not found")

        return explainer(**kwargs)
    
    _cache = {}

    @staticmethod
    def get_by_name(explainer_name: str) -> BaseExplainer:
        """
        Load an explainer from a saved file, utilizing an in-memory cache.

        Parameters
        ----------
        explainer_name : str
            The name of the explainer to load.

        Returns
        -------
        BaseExplainer
            The loaded explainer.

        Raises
        ------
        FileNotFoundError
            If the explainer file is not found.
        """
        if not MODELS_DIR.exists() or not (MODELS_DIR / f"{explainer_name}.pth").exists():
            raise FileNotFoundError(f"Explainer {explainer_name} not found on disk.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if explainer_name not in BaseExplainer._cache:
            logger.info(f"Explainer '{explainer_name}' not in cache. Loading from disk...")
            explainer = torch.load(MODELS_DIR / f"{explainer_name}.pth", weights_only=False, map_location=device)
            explainer.name = explainer_name
            BaseExplainer._cache[explainer_name] = explainer
            logger.info(f"Explainer '{explainer_name}' successfully loaded and cached.")
        else:
            logger.info(f"Explainer '{explainer_name}' retrieved from cache.")

        return BaseExplainer._cache[explainer_name]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method_name!r})"
