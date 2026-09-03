"""
core/utils/metrics.py
=======================
This module provides the functionality for assessing hardware and XAI metrics
for each run of the detectors and explainers.

Hardware Metrics:
- Time taken (elapsed seconds)
- Peak memory usage (MB)
(Implemented via the @track_hardware decorator)

XAI Metrics:
- Sparsity
- Faithfulness
- Stability
- Consistency
(Implemented as pure functions for evaluation)
"""

import time
import tracemalloc
from functools import wraps
from typing import Any, Callable

import torch

from core.results.base import Result

# =========================================================================
#  HARDWARE METRICS
# =========================================================================


def track_hardware(func: Callable) -> Callable:
    """
    A decorator that wraps a function (like predict() or explain())
    to measure its execution time and peak RAM usage.
    
    If the wrapped function returns a `Result` object (or subclass),
    this decorator automatically injects the `elapsed_seconds` and
    `peak_ram_mb` properties into the result.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        tracemalloc.start()
        t0 = time.perf_counter()
        
        # Execute the actual AI model/explainer method
        result = func(*args, **kwargs)
        
        # Capture metrics
        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # If it's our Result DTO, inject the metrics
        if isinstance(result, Result):
            result.metrics["elapsed_seconds"] = round(elapsed, 4)
            result.metrics["peak_ram_mb"] = round(peak_mem / (1024 * 1024), 2)
            
        return result
        
    return wrapper


# =========================================================================
#  XAI METRICS (Stubs)
# =========================================================================

def compute_sparsity(heatmap: torch.Tensor, threshold: float = 0.05) -> float:
    """
    Measures how much of the heatmap is zero (or near zero).
    Higher sparsity means the explanation is more concise and focused.
    """
    if not isinstance(heatmap, torch.Tensor):
        heatmap = torch.tensor(heatmap)
    
    # Normalize heatmap to [0, 1] for thresholding
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max > h_min:
        h_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        h_norm = heatmap
    
    # Calculate fraction of pixels below threshold
    sparsity = (h_norm < threshold).float().mean().item()
    return round(sparsity, 4)


def compute_faithfulness(model, image_tensor: torch.Tensor, heatmap: torch.Tensor, target_class: int = None) -> float:
    """
    Measures if removing the 'important' pixels (as defined by the heatmap)
    actually drops the model's confidence in its original prediction.
    Highly faithful XAI means the heatmap accurately identifies the model's logic.

    Occlusion fill strategy: Gaussian blur (sigma=10) instead of constant grey.
    Constant grey (0.5) is out-of-distribution for models trained on face images
    and can trigger artefact responses unrelated to the missing content.
    Blurring removes high-frequency detail (textures, edges) while keeping
    low-frequency content (colours, rough shapes) in-distribution.
    """
    if not isinstance(heatmap, torch.Tensor):
        heatmap = torch.tensor(heatmap)

    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Base prediction confidence
        base_outputs = model(image_tensor)
        base_probs = torch.softmax(base_outputs, dim=1)
        if target_class is None:
            target_class = base_probs.argmax(dim=1).item()
        base_conf = base_probs[0, target_class].item()

        # Build occlusion mask: zero out top 40% most important pixels
        h_flat = heatmap.flatten()
        k = max(1, int(0.4 * h_flat.numel()))
        top_indices = torch.topk(h_flat, k).indices
        mask_flat = torch.ones_like(h_flat)
        mask_flat[top_indices] = 0.0
        mask = mask_flat.view(heatmap.shape)

        # Expand mask to [1, C, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = mask.to(device)

        # Precompute blurred fill (in-distribution replacement for occluded pixels)
        from torchvision.transforms.functional import gaussian_blur as _gb
        _, _, h, w = image_tensor.shape
        k_blur = max(11, (min(h, w) // 3) | 1)   # always odd, >= 11
        blurred = _gb(
            image_tensor.squeeze(0), kernel_size=k_blur, sigma=10.0
        ).unsqueeze(0).to(device)

        # Replace important pixels with their blurred counterparts
        occluded_image = image_tensor * mask + blurred * (1 - mask)

        # Confidence after occlusion
        occ_outputs = model(occluded_image)
        occ_probs = torch.softmax(occ_outputs, dim=1)
        occ_conf = occ_probs[0, target_class].item()

    faithfulness = base_conf - occ_conf
    return round(faithfulness, 4)


import tempfile
from pathlib import Path

def compute_stability(explainer, image_path: str, target_class: int, detector, noise_level: float = 0.05, runs: int = 5) -> float:
    """
    Measures if small, imperceptible perturbations to the input image
    drastically change the explanation. Good XAI should be stable.
    Now using Pearson correlation to measure structural similarity instead of MSE.
    """
    # Get base heatmap
    base_result = explainer.explain(detector, image_path, target_class)
    base_heatmap = base_result.returned_obj
    base_h = base_heatmap.detach().float()
        
    image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)
    
    correlations = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "noisy.pt"
        for _ in range(runs):
            # Add Gaussian noise
            noise = torch.randn_like(image_tensor) * noise_level
            noisy_image = image_tensor + noise
            torch.save(noisy_image, tmp_path)
            
            # Generate new heatmap
            noisy_result = explainer.explain(detector, str(tmp_path), target_class)
            noisy_heatmap = noisy_result.returned_obj
            noisy_h = noisy_heatmap.detach().float()
                
            # Calculate Pearson Correlation (reusing consistency function)
            corr = compute_consistency(base_h, noisy_h)
            correlations.append(corr)
        
    # Stability is the mean correlation across noisy runs
    stability = sum(correlations) / len(correlations)
    return round(stability, 4)


def compute_consistency(heatmap_a: torch.Tensor, heatmap_b: torch.Tensor) -> float:
    """
    Measures the similarity between two different explainers applied to the
    same image and model. E.g., do Grad-CAM and Occlusion agree?
    """
    # Ensure they are tensors
    if not isinstance(heatmap_a, torch.Tensor):
        heatmap_a = torch.tensor(heatmap_a)
    if not isinstance(heatmap_b, torch.Tensor):
        heatmap_b = torch.tensor(heatmap_b)
        
    a_flat = heatmap_a.flatten().float()
    b_flat = heatmap_b.flatten().float()
    
    # Calculate Pearson Correlation Coefficient
    # Covariance(A, B) / (StdDev(A) * StdDev(B))
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    
    cov = ((a_flat - a_mean) * (b_flat - b_mean)).mean()
    std_a = a_flat.std(unbiased=False)
    std_b = b_flat.std(unbiased=False)
    
    # Avoid division by zero
    if std_a == 0 or std_b == 0:
        return 0.0
        
    correlation = (cov / (std_a * std_b)).item()
    return round(correlation, 4)
