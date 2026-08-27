"""
core/xai/explainers/gradient.py
================================
Gradient-based XAI explainers.

These methods trace gradients backward through the network to determine
which parts of the image (or feature map) were most responsible for the
model's prediction.

They are typically "white-box" methods because they require access to
the model's internal architecture (e.g., target layers) and the ability
to backpropagate.
"""

from __future__ import annotations

from core.utils.metrics import track_hardware

import torch
import torch.nn.functional as F
from pathlib import Path

from .base import BaseExplainer
from core.results.explanation import XAIResult
from core.detectors.base import AbstractBaseDetector
from core.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAMExplainer(BaseExplainer):
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    WHAT IT DOES
    -------------
    Grad-CAM highlights the regions of the image that the model looked
    at to make its decision. It does this by looking at the LAST
    convolutional layer (the highest-level features just before the
    classifier) and checking which feature maps were activated.

    HOW IT WORKS (Math)
    --------------------
    1. Forward pass: compute logits for the image.
    2. Backward pass: compute gradients of the TARGET CLASS logit
       with respect to the target conv layer's feature maps (A^k).
    3. Global Average Pooling (GAP) of gradients to get neuron
       importance weights: alpha_k = GAP( dY / dA^k ).
    4. Weighted combination: Heatmap = ReLU( sum(alpha_k * A^k) ).
    5. Resize heatmap to the original image dimensions.

    WHY IT'S GOOD
    -------------
    It is class-discriminative (it shows what evidence supports a
    SPECIFIC class) and computationally cheap (only one forward and
    one backward pass).

    LIMITATIONS
    -----------
    It fundamentally relies on convolutional layers. For ViTs, we use
    Attention Rollout instead (though Grad-CAM can technically be
    applied to the ViT's patch embedding projection as a fallback).
    """

    method_name = "grad_cam"

    @track_hardware
    def explain(
        self,
        detector: AbstractBaseDetector,
        image_path: str,
        target_class: int,
    ) -> XAIResult:

        logger.info(f"[Grad-CAM] Starting explanation for {image_path} with target_class={target_class}")
        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False).clone()
        
        # 2. Get the target layer
        target_layer = detector.get_target_layer()
        if target_layer is None:
            raise ValueError(
                f"Detector {detector.name} does not support Grad-CAM "
                "(no target layer defined)."
            )

        device = next(detector.parameters()).device
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)
        _, _, h, w = x.shape

        # 3. Register hooks to capture forward activations and backward gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        hook_f = target_layer.register_forward_hook(forward_hook)
        hook_b = target_layer.register_full_backward_hook(backward_hook)

        # 4. Forward pass
        detector.eval()
        detector.zero_grad()
        logits = detector(x)
        
        # Capture basic prediction stats
        probs = F.softmax(logits, dim=1).squeeze(0)
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        # 5. Backward pass for the target class
        target_logit = logits[0, target_class]
        target_logit.backward()

        # 6. Remove hooks
        hook_f.remove()
        hook_b.remove()

        # 7. Compute Grad-CAM
        # gradients[0] -> [1, C, H', W']
        # activations[0] -> [1, C, H', W']
        if not gradients or not activations:
            raise RuntimeError("Hooks failed to capture gradients/activations.")

        grad = gradients[0][0]  # [C, H', W']
        act = activations[0][0] # [C, H', W']

        # Global average pool gradients -> weights [C, 1, 1]
        weights = grad.mean(dim=(1, 2), keepdim=True)

        # Weighted sum of activations -> [H', W']
        cam = (weights * act).sum(dim=0)

        # Apply ReLU to only keep features that have a POSITIVE influence
        cam = F.relu(cam)

        # 8. Resize and normalize
        cam = cam.unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        logger.info(f"[Grad-CAM] Heatmap successfully generated. Dimensions: {cam.shape}")

        return XAIResult(
            ai_deepfake=(class_idx == 1),
            confidence=confidence,
            image=image_path,
            returned_obj=cam,
            explainer=self,
            forward_passes=1,  # one forward, one backward
        )


class VanillaSaliencyExplainer(BaseExplainer):
    """
    Vanilla Gradient Saliency.

    WHAT IT DOES
    -------------
    Highlights the pixels that are most sensitive to change. It asks:
    "If I slightly change this pixel, how much does the probability of
    the target class change?"

    HOW IT WORKS (Math)
    --------------------
    1. Forward pass: compute logits for the image.
    2. Backward pass: compute gradients of the TARGET CLASS logit
       with respect to the INPUT PIXELS.
       Grad = dY / dX
    3. Take the absolute value (magnitude matters, sign doesn't).
    4. Take the maximum across color channels (if RGB) to get a
       single 2D heatmap.

    WHY IT'S GOOD
    -------------
    Extremely simple, fast, and works on ANY differentiable model
    (CNN, ViT, MLP) without needing to hook into specific layers.

    LIMITATIONS
    -----------
    Heatmaps are often noisy and highlight high-frequency edges rather
    than semantic objects (this is known as the "shattered gradients"
    problem). It's more of a sanity check baseline than a state-of-the-art
    explainer.
    """

    method_name = "vanilla_saliency"

    @track_hardware
    def explain(
        self,
        detector: AbstractBaseDetector,
        image_path: str,
        target_class: int,
    ) -> XAIResult:

        logger.info(f"[Vanilla Saliency] Starting explanation for {image_path} with target_class={target_class}")
        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)
        
        device = next(detector.parameters()).device
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)

        # We need gradients with respect to the input image
        x.requires_grad_()

        detector.eval()
        detector.zero_grad()

        # Forward pass
        logits = detector(x)
        
        # Capture basic prediction stats
        probs = F.softmax(logits, dim=1).squeeze(0)
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        # Backward pass
        target_logit = logits[0, target_class]
        target_logit.backward()

        # Extract gradients
        gradients = x.grad[0]  # [C, H, W]

        # Take absolute value and max across color channels
        # [C, H, W] -> [H, W]
        saliency_map, _ = torch.max(gradients.abs(), dim=0)

        # Normalize to [0, 1]
        saliency_map = saliency_map.cpu().detach()
        s_min, s_max = saliency_map.min(), saliency_map.max()
        if s_max - s_min > 1e-8:
            saliency_map = (saliency_map - s_min) / (s_max - s_min)
        else:
            saliency_map = torch.zeros_like(saliency_map)

        logger.info(f"[Vanilla Saliency] Heatmap successfully generated. Dimensions: {saliency_map.shape}")

        return XAIResult(
            ai_deepfake=(class_idx == 1),
            confidence=confidence,
            image=image_path,
            returned_obj=saliency_map,
            explainer=self,
            forward_passes=1,
        )
