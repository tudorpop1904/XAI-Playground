"""
core/detectors/base.py
=======================
Abstract base class for all deepfake image detectors.

Detectors are responsible for:
  1. Defining the neural network architecture (nn.Module)
  2. Running inference on a single image → DetectionResult
  3. Exposing hooks (like the target convolutional layer) so that
     external explainer services can attach to them for XAI.

Detectors are NOT responsible for:
  - XAI/explanation logic (that lives in core/xai/explainers/)
  - Database persistence (that lives in db/)
  - API serialisation (that lives in api/schemas/)

This separation follows the Single Responsibility Principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from core.utils.paths import MODELS_DIR
from core.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from core.results.detection import DetectionResult


class AbstractBaseDetector(ABC, nn.Module):
    """
    Base class for all image detectors used by the application.

    Every concrete detector (CNNDetector, KNNDetector, ViTDetector)
    must implement:
      - forward()   — the standard PyTorch forward pass
      - predict()   — single image inference returning a DetectionResult

    Optionally, detectors can override:
      - get_target_layer() — returns the conv layer that Grad-CAM should
                             hook into (default: None)
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Raw logits of shape [B, num_classes].
        """
        pass

    @staticmethod
    def get_detector(detector_type: str) -> AbstractBaseDetector:
        """
        Instantiate a detector by type.

        Parameters
        ----------
        detector_type : str
            The type of detector to instantiate.

        Returns
        -------
        AbstractBaseDetector
            The instantiated detector.

        Raises
        ------
        ValueError
            If the detector type is not found.
        """
        from core.detectors.cnn import CNNDetector
        from core.detectors.vit import ViTDetector
        from core.detectors.knn import KNNDetector

        DETECTORS = {
            "CNN": CNNDetector,
            "VIT": ViTDetector,
            "KNN": KNNDetector
        }

        model = DETECTORS.get(detector_type.upper())

        if model is None:
            raise ValueError("Detector type not found")

        return model
    
    _cache = {}

    @staticmethod
    def get_by_name(model_name: str) -> AbstractBaseDetector:
        """
        Load a detector from a saved file, utilizing an in-memory cache.

        Parameters
        ----------
        model_name : str
            The name of the detector to load.

        Returns
        -------
        AbstractBaseDetector
            The loaded detector.

        Raises
        ------
        FileNotFoundError
            If the detector file is not found.
        """
        if not MODELS_DIR.exists() or not (MODELS_DIR / f"{model_name}.pth").exists():
            raise FileNotFoundError(f"Model {model_name} not found on disk.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name not in AbstractBaseDetector._cache:
            logger.info(f"Model '{model_name}' not in cache. Loading from disk...")
            model = torch.load(MODELS_DIR / f"{model_name}.pth", weights_only=False, map_location=device)
            model.name = model_name
            AbstractBaseDetector._cache[model_name] = model
            logger.info(f"Model '{model_name}' successfully loaded and cached.")
        else:
            logger.info(f"Model '{model_name}' retrieved from cache.")

        return AbstractBaseDetector._cache[model_name]

    @abstractmethod
    def predict(self, image_tensor: torch.Tensor) -> DetectionResult:
        """
        Run inference on a single image and return a structured result.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image tensor of shape [C, H, W], values in [0, 1].

        Returns
        -------
        DetectionResult
            Contains the predicted label, confidence, and class
            probabilities.
        """
        pass

    def get_target_layer(self) -> Optional[nn.Module]:
        """
        Return the convolutional layer that XAI methods (e.g. Grad-CAM)
        should hook into.

        By default returns None. Concrete detectors should override this
        to point to their last conv layer (or whichever layer is most
        meaningful for explanation).

        Returns
        -------
        nn.Module or None
            The target layer, or None if the detector does not support
            layer-based explanations.
        """
        return None