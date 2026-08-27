"""
core/explainers/__init__.py
============================
Unified visual XAI for AI-generated image detection.

This module acts as the Orchestrator/Factory for all XAI explainers.
It provides a central registry where the API and UI layers can request
a specific explanation method without needing to know its class details.

AVAILABLE METHODS
------------------
White-box (requires model architecture + gradients):
  - Grad-CAM (gradient-weighted class activation mapping)
  - Saliency (vanilla input gradients)

Black-box (model-agnostic, perturbation-based):
  - Occlusion Sensitivity (slide a grey box)
  - Visual PMI (reveal only one box)
  - Visual Sobol (Monte Carlo variance reduction)
"""

from __future__ import annotations

from .base import BaseExplainer
from .gradient import GradCAMExplainer, VanillaSaliencyExplainer
from .perturbation import OcclusionExplainer, SobolExplainer, PMIExplainer

# =========================================================================
#  REGISTRY
# =========================================================================
#
# This maps string identifiers to the corresponding explainer classes.
# To add a new XAI method in the future, simply write the class in
# the explainers/ folder and add it to this dictionary.
# =========================================================================

EXPLAINER_REGISTRY: dict[str, type[BaseExplainer]] = {
    GradCAMExplainer.method_name: GradCAMExplainer,
    VanillaSaliencyExplainer.method_name: VanillaSaliencyExplainer,
    OcclusionExplainer.method_name: OcclusionExplainer,
    PMIExplainer.method_name: PMIExplainer,
    SobolExplainer.method_name: SobolExplainer,
}


def get_explainer(method_name: str, **kwargs) -> BaseExplainer:
    """
    Factory method to instantiate an explainer by name.

    Parameters
    ----------
    method_name : str
        The unique identifier for the explainer (e.g., "grad_cam").
    **kwargs : dict
        Optional configuration parameters to pass to the explainer's
        constructor (e.g., grid_rows=8, n_samples=128).

    Returns
    -------
    BaseExplainer
        An instantiated explainer ready to run `.explain()`.

    Raises
    ------
    ValueError
        If the requested method_name is not in the registry.
    """
    if method_name not in EXPLAINER_REGISTRY:
        available = ", ".join(EXPLAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown XAI method: '{method_name}'. "
            f"Available methods are: {available}"
        )
    
    explainer_cls = EXPLAINER_REGISTRY[method_name]
    return explainer_cls(**kwargs)


def list_available_methods() -> list[str]:
    """
    Return a list of all registered XAI method names.
    Useful for populating UI dropdowns.
    """
    return list(EXPLAINER_REGISTRY.keys())
