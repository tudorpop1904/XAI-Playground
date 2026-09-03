"""
core/xai/explainers/perturbation.py
====================================
Perturbation-based XAI explainers (Black-Box).

These methods treat the model as a black box. They don't need access to
gradients or internal layers. Instead, they divide the image into a grid,
occlude or reveal cells, and measure how the model's output probability
changes. 

This is computationally expensive (requires many forward passes) but
produces highly robust and intuitive explanations that are model-agnostic.
"""

from __future__ import annotations

from datetime import datetime

from core.utils.metrics import track_hardware

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseExplainer
from core.results.explanation import XAIResult
from core.detectors.base import AbstractBaseDetector


# =========================================================================
#  UTILITIES
# =========================================================================

def _apply_grid_mask(
    x_tensor: torch.Tensor,
    mask: np.ndarray,
    grid_rows: int,
    grid_cols: int,
    fill,          # float OR torch.Tensor of same shape as x_tensor
    mode: str,
) -> torch.Tensor:
    """
    Apply a low-res boolean mask to a high-res image tensor.

    mode='occlude': replace cells where mask==0 with fill (keep mask==1 visible).
    mode='reveal':  keep only cells where mask==1; replace rest with fill.

    fill can be:
      - float  : constant fill value (e.g. 0.5 for grey)
      - Tensor : same shape as x_tensor, e.g. a precomputed blurred image.
                 Each cell region is replaced with the corresponding region
                 of the fill tensor, keeping the fill spatially coherent.
    """
    masked = x_tensor.clone()
    _, c, h, w = x_tensor.shape
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    use_tensor_fill = isinstance(fill, torch.Tensor)

    for r in range(grid_rows):
        for c_idx in range(grid_cols):
            active = mask[r, c_idx] == 1
            if (mode == "occlude" and active) or (mode == "reveal" and not active):
                continue

            r0 = r * cell_h
            r1 = (r + 1) * cell_h if r < grid_rows - 1 else h
            c0 = c_idx * cell_w
            c1 = (c_idx + 1) * cell_w if c_idx < grid_cols - 1 else w

            if use_tensor_fill:
                masked[:, :, r0:r1, c0:c1] = fill[:, :, r0:r1, c0:c1]
            else:
                masked[:, :, r0:r1, c0:c1] = fill

    return masked


def _make_blur_fill(x: torch.Tensor) -> torch.Tensor:
    """
    Precompute a strongly-blurred version of x for use as occlusion fill.

    Gaussian blur with sigma=10 eliminates high-frequency detail (edges,
    textures) while preserving low-frequency content (overall colours,
    rough shapes). This keeps the occluded image in-distribution for
    models trained on natural/generated face images, unlike a constant
    grey value which the model has never seen during training.

    The kernel size is chosen as ~1/3 of the shortest spatial dimension,
    rounded up to the nearest odd integer (required by torchvision).
    """
    from torchvision.transforms.functional import gaussian_blur as _gb
    _, _, h, w = x.shape
    k = max(11, (min(h, w) // 3) | 1)   # always odd, at least 11
    # gaussian_blur expects [C, H, W] or [..., C, H, W]
    blurred = _gb(x.squeeze(0), kernel_size=k, sigma=10.0).unsqueeze(0)
    return blurred.to(x.device)


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    """Min-max normalization of a 2D grid to [0, 1]."""
    smin, smax = float(grid.min()), float(grid.max())
    if smax - smin > 1e-8:
        return (grid - smin) / (smax - smin)
    return np.zeros_like(grid)


def _upscale_grid(grid: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Upscale a low-res [rows, cols] grid to a smooth [H, W] heatmap."""
    heatmap_pil = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    heatmap_pil = heatmap_pil.resize((w, h), Image.BILINEAR)
    arr = np.asarray(heatmap_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _get_prob(detector: AbstractBaseDetector, x: torch.Tensor, target_class: int) -> float:
    """Helper to do a quick forward pass and get the probability of a class."""
    with torch.inference_mode():
        logits = detector(x)
        probs = F.softmax(logits, dim=1)
        return probs[0, target_class].item()


# =========================================================================
#  EXPLAINERS
# =========================================================================

class OcclusionExplainer(BaseExplainer):
    """
    Occlusion Sensitivity.

    WHAT IT DOES
    -------------
    Slides a grey box over the image and measures how much the
    confidence DROPS. If the confidence drops heavily when a specific
    region is hidden, that region was highly important.

    MATH
    ----
    Score(r, c) = max(0, P_baseline - P_occluded(r, c))
    """

    method_name = "occlusion_sensitivity"

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4, fill: float = 0.5):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fill = fill

    @track_hardware
    def explain(
        self,
        detector: AbstractBaseDetector,
        image_path: str,
        target_class: int,
    ) -> XAIResult:
        
        from pathlib import Path
        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)
        device = next(detector.parameters()).device
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)
        _, _, h, w = x.shape
        total_cells = self.grid_rows * self.grid_cols

        detector.eval()
        
        # Get baseline prediction stats
        with torch.inference_mode():
            logits = detector(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            class_idx = probs.argmax().item()
            confidence = probs[class_idx].item()
            baseline_prob = probs[target_class].item()

        # Precompute blurred fill once — kept in-distribution vs. constant grey
        blur_fill = _make_blur_fill(x)

        scores = np.zeros((self.grid_rows, self.grid_cols))
        cell_scores = {}

        for idx in range(total_cells):
            r, c = idx // self.grid_cols, idx % self.grid_cols
            mask = np.ones((self.grid_rows, self.grid_cols), dtype=np.int8)
            mask[r, c] = 0  # Occlude this cell

            occluded = _apply_grid_mask(
                x, mask, self.grid_rows, self.grid_cols, blur_fill, mode="occlude"
            )
            p_occ = _get_prob(detector, occluded, target_class)

            drop = max(0.0, baseline_prob - p_occ)
            scores[r, c] = drop
            cell_scores[f"r{r}_c{c}"] = drop

        grid = _normalize_grid(scores)
        heatmap = _upscale_grid(grid, h, w)

        return XAIResult(
            ai_deepfake=(class_idx == 1),
            confidence=confidence,
            image=image_path,
            returned_obj=heatmap,
            explainer=self,
            forward_passes=total_cells + 1,
            cell_scores=cell_scores,
        )


class PMIExplainer(BaseExplainer):
    """
    Visual Pointwise Mutual Information (PMI).

    WHAT IT DOES
    -------------
    Instead of hiding one cell, we HIDE EVERYTHING EXCEPT one cell.
    We measure the probability of the target class when only that
    cell is visible, compared to a baseline where everything is grey.

    MATH
    ----
    Score(r, c) = log2( P_revealed(r, c) / P_grey )
    """

    method_name = "pmi"

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    @track_hardware
    def explain(
        self,
        detector: AbstractBaseDetector,
        image_path: str,
        target_class: int,
    ) -> XAIResult:
        
        from pathlib import Path
        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)
        device = next(detector.parameters()).device
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)
        _, _, h, w = x.shape
        total_cells = self.grid_rows * self.grid_cols

        detector.eval()
        
        with torch.inference_mode():
            logits = detector(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            class_idx = probs.argmax().item()
            confidence = probs[class_idx].item()

        # Baseline: fully-blurred image ("no fine detail" reference).
        # Using blur rather than constant grey keeps the baseline in-distribution
        # while still representing the minimum-information state for PMI.
        blur_fill = _make_blur_fill(x)
        p_baseline = _get_prob(detector, blur_fill, target_class)

        scores = np.zeros((self.grid_rows, self.grid_cols))
        cell_scores = {}

        for idx in range(total_cells):
            r, c = idx // self.grid_cols, idx % self.grid_cols
            mask = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int8)
            mask[r, c] = 1  # Reveal only this cell

            # Hidden cells show blurred content; only (r,c) shows original pixels
            revealed = _apply_grid_mask(
                x, mask, self.grid_rows, self.grid_cols, blur_fill, mode="reveal"
            )
            p_reveal = _get_prob(detector, revealed, target_class)

            ratio = (p_reveal + 1e-7) / (p_baseline + 1e-7)
            score = float(np.log2(ratio))

            # Keep only positive information
            score = max(0.0, score)
            scores[r, c] = score
            cell_scores[f"r{r}_c{c}"] = score

        grid = _normalize_grid(scores)
        heatmap = _upscale_grid(grid, h, w)

        return XAIResult(
            ai_deepfake=(class_idx == 1),
            confidence=confidence,
            image=image_path,
            returned_obj=heatmap,
            explainer=self,
            forward_passes=total_cells + 2,
            cell_scores=cell_scores,
        )


class SobolExplainer(BaseExplainer):
    """
    Monte Carlo Variance Reduction (Sobol).

    WHAT IT DOES
    -------------
    Generates N random binary masks (some cells occluded, some visible).
    Passes all N masked images through the model, getting N probabilities.
    For each cell, it compares the total variance across all N masks to
    the variance of the subset of masks where that specific cell was visible.

    If fixing a cell to be visible drastically reduces the variance
    of the model's output, it means that cell exerts a strong influence
    over the model's decision.

    MATH
    ----
    Score(r, c) = (Var(Total) - Var(Subset where r,c is visible)) / Var(Total)
    """

    method_name = "sobol"

    def __init__(self, grid_rows: int = 4, grid_cols: int = 4, n_samples: int = 64, seed: int = 42):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_samples = n_samples
        self.seed = seed

    @track_hardware
    def explain(
        self,
        detector: AbstractBaseDetector,
        image_path: str,
        target_class: int,
    ) -> XAIResult:
        
        from pathlib import Path
        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)
        device = next(detector.parameters()).device
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)
        _, _, h, w = x.shape
        
        rng = np.random.default_rng(self.seed)

        detector.eval()
        
        with torch.inference_mode():
            logits = detector(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            class_idx = probs.argmax().item()
            confidence = probs[class_idx].item()

        # Precompute blurred fill once — used for all N masked forward passes
        blur_fill = _make_blur_fill(x)

        masks: list[np.ndarray] = []
        predictions: list[float] = []

        for _ in range(self.n_samples):
            # 50% chance for each cell to be visible/occluded
            mask = rng.binomial(1, 0.5, size=(self.grid_rows, self.grid_cols)).astype(np.int8)

            # Avoid completely empty masks
            if mask.sum() == 0:
                mask[rng.integers(self.grid_rows), rng.integers(self.grid_cols)] = 1

            masked = _apply_grid_mask(
                x, mask, self.grid_rows, self.grid_cols, blur_fill, mode="occlude"
            )
            p = _get_prob(detector, masked, target_class)

            predictions.append(p)
            masks.append(mask)

        var_total = float(np.var(predictions))
        scores = np.zeros((self.grid_rows, self.grid_cols))
        cell_scores = {}

        if var_total > 1e-8:
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    # Filter predictions where this specific cell was visible
                    subset = [p for idx, p in enumerate(predictions) if masks[idx][r, c] == 1]
                    
                    if len(subset) > 1:
                        var_cond = float(np.var(subset))
                        score = max(0.0, (var_total - var_cond) / var_total)
                    else:
                        score = 0.0
                        
                    scores[r, c] = score
                    cell_scores[f"r{r}_c{c}"] = score

        grid = _normalize_grid(scores)
        heatmap = _upscale_grid(grid, h, w)

        return XAIResult(
            ai_deepfake=(class_idx == 1),
            confidence=confidence,
            image=image_path,
            returned_obj=heatmap,
            explainer=self,
            forward_passes=self.n_samples + 1,
            cell_scores=cell_scores,
        )
