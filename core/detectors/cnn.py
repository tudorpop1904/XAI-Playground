"""
core/detectors/cnn.py
======================
FTL (Faster-Than-Lies) CNN Deepfake Detector
---------------------------------------------

This module implements the core CNN architecture from the FTL deepfake
detection methodology. It is the primary detector in the application:
a from-scratch convolutional neural network that classifies images as
either REAL or AI-GENERATED.

ARCHITECTURE OVERVIEW
----------------------
The FTL CNN differs from a standard image classifier in one critical
way: it receives not just RGB, but also forensic feature channels
(FFT, LBP, Sobel) computed by core/xai/image_features.py.

The full pipeline is:

    Input image [3, H, W]
         │
         ▼
    Forensic Feature Engineering  (if enabled)
    [3, H, W] → [3+K, H, W]   where K ∈ {0, 1, 2, 3}
         │
         ▼
    1×1 Channel Adapter
    [3+K, H, W] → [3, H, W]   (learned channel mixing)
         │
         ▼
    4 Convolutional Blocks
    Each: Conv2d → BatchNorm → ReLU → MaxPool2d
    [3, H, W] → [32, H/2, W/2] → [64, H/4, W/4]
              → [128, H/8, W/8] → [256, H/16, W/16]
         │
         ▼
    Global Average Pooling
    [256, H/16, W/16] → [256, 1, 1] → [256]
         │
         ▼
    Dropout (p=0.5)
         │
         ▼
    Fully Connected Head
    [256] → [num_classes]   (raw logits)

PAPER REFERENCE
----------------
• R. Lanzino, F. Fontana, A. Diko, M. R. Marini, L. Cinque (CVPRW 2024).
  "Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks"
• Paper: https://iris.uniroma1.it/retrieve/59badfa8-a0da-4a27-b85c-fad48d771c23/
         Lanzino_postprint_Faster_2024.pdf
• Repository: https://github.com/fedeloper/binary_deepfake_detection
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from core.utils.metrics import track_hardware
from torch import nn
from torch.utils.data import DataLoader

from .base import AbstractBaseDetector
from core.utils.image_features import compute_fft, compute_lbp, compute_sobel

from .base import logger

if TYPE_CHECKING:
    from core.results.detection import DetectionResult


# ── Device-agnostic setup ───────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Class label mapping ─────────────────────────────────────────────
CLASS_NAMES = {0: "Real", 1: "AI-Generated"}


class CNNDetector(AbstractBaseDetector):
    """
    FTL-style CNN (NOT BNN) for deepfake detection.

    This detector can optionally prepend forensic channels (FFT, LBP,
    Sobel) to the RGB input, then uses a 1×1 convolution to mix them
    back down to 3 channels before passing through 4 standard
    convolutional blocks.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 2 — Real vs AI-Generated).
    add_fft : bool
        If True, prepend the FFT magnitude channel. (default: False)
    add_lbp : bool
        If True, prepend the LBP texture channel. (default: False)
    add_sobel : bool
        If True, prepend the Sobel edge magnitude channel. (default: False)
    dropout_rate : float
        Dropout probability before the final FC layer (default: 0.5).

    Example
    -------
    >>> model = CNNDetector(add_fft=True, add_lbp=True, add_sobel=True)
    >>> x = torch.randn(1, 3, 128, 128)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([1, 2])
    """

    # ── Class-level counter for auto-naming instances ────────────────
    index = 1

    def __init__(
        self,
        num_classes: int = 2,
        add_fft: bool = False,
        add_lbp: bool = False,
        add_sobel: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__(name=f"CNN_{CNNDetector.index}")
        CNNDetector.index += 1

        # Store configuration for serialisation / logging
        self.num_classes = num_classes
        self.add_fft = add_fft
        self.add_lbp = add_lbp
        self.add_sobel = add_sobel
        self.dropout_rate = dropout_rate

        # ────────────────────────────────────────────────────────────
        # STAGE 0: Forensic Channel Adapter (1×1 Convolution)
        # ────────────────────────────────────────────────────────────
        #
        # If forensic channels are enabled, the input has more than 3
        # channels (up to 6). The 1×1 conv learns how to MIX the extra
        # channels with RGB to produce 3 "super-channels".
        #
        # MATH:
        #   For each pixel (h, w), the adapter computes:
        #
        #     super_ch_i(h,w) = Σ_{j=1}^{C_in} w_{ij} · channel_j(h,w)
        #
        #   where C_in = 3 + K extra forensic channels.
        #
        #   This is equivalent to a per-pixel fully-connected layer
        #   across the channel dimension — no spatial context is used,
        #   only the channel values at that specific pixel.
        #
        # WHY bias=False?
        #   The next layer (BatchNorm) already has a learnable bias (β).
        #   Adding bias here would be redundant: the optimiser would
        #   just learn to cancel one against the other.
        #
        # WHY nn.Identity() when no extras?
        #   If only RGB is used (K=0), we skip the 1×1 conv entirely.
        #   nn.Identity() passes the input through unchanged, so the
        #   rest of the forward() logic stays the same.
        # ────────────────────────────────────────────────────────────

        extra_channels = sum([self.add_fft, self.add_lbp, self.add_sobel])
        self.input_channels = 3 + extra_channels

        if extra_channels > 0:
            self.feature_adapter = nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=3,
                kernel_size=1,
                bias=False,
            )
        else:
            self.feature_adapter = nn.Identity()

        # ────────────────────────────────────────────────────────────
        # STAGE 1–4: Convolutional Blocks
        # ────────────────────────────────────────────────────────────
        #
        # Each block follows the pattern:
        #   Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)
        #
        # CONV2D: Applies a learned 3×3 kernel to extract spatial
        #   features. padding=1 ensures same-size output before pool.
        #
        #   Output shape: [B, C_out, H, W]  (same H, W as input)
        #
        #   Parameter count per conv: C_in × C_out × 3 × 3
        #   Example: Conv2d(3, 32, 3) has 3 × 32 × 9 = 864 weights.
        #
        # BATCHNORM2D: Normalises each channel's activations across
        #   the batch to have mean=0, std=1, then applies a learnable
        #   affine transform (γ, β). This:
        #     - Stabilises training (prevents internal covariate shift)
        #     - Acts as a regulariser (noise from batch statistics)
        #     - Allows higher learning rates
        #
        #   MATH:  x̂ = (x - μ_batch) / √(σ²_batch + ε)
        #          y = γ · x̂ + β
        #
        #   During training: μ and σ² are computed per mini-batch.
        #   During inference: μ and σ² are running averages from training.
        #
        # RELU: f(x) = max(0, x)
        #   Introduces non-linearity. Without it, stacking conv layers
        #   would be equivalent to a single conv (linear composition of
        #   linear functions is linear). ReLU is preferred over sigmoid
        #   because:
        #     - No vanishing gradient for positive values (gradient = 1)
        #     - Computationally cheap (just a threshold)
        #     - Encourages sparse activations (many zeros)
        #
        # MAXPOOL2D(2): Takes the maximum value in each 2×2 window.
        #   Halves spatial dimensions: [H, W] → [H/2, W/2].
        #   This provides:
        #     - Translation invariance (a feature shifted by 1 pixel
        #       is still detected)
        #     - Receptive field growth (each subsequent conv "sees"
        #       a larger area of the original image)
        #     - Reduced computation (fewer pixels to process)
        #
        # SHAPE PROGRESSION (for 128×128 input):
        #   Block 1: [B, 3,  128, 128] → [B, 32,  64, 64]
        #   Block 2: [B, 32,  64,  64] → [B, 64,  32, 32]
        #   Block 3: [B, 64,  32,  32] → [B, 128, 16, 16]
        #   Block 4: [B, 128, 16,  16] → [B, 256,  8,  8]
        #
        # WHY 4 BLOCKS?
        #   Each block doubles channels and halves spatial size. After
        #   4 blocks, a 128×128 image is reduced to 8×8 feature maps
        #   with 256 channels — enough spatial reduction for GAP to
        #   compress to a meaningful vector, while retaining enough
        #   channels to encode diverse features.
        # ────────────────────────────────────────────────────────────

        self.block1 = self._build_conv_block(in_ch=3,   out_ch=32)
        self.block2 = self._build_conv_block(in_ch=32,  out_ch=64)
        self.block3 = self._build_conv_block(in_ch=64,  out_ch=128)
        self.block4 = self._build_conv_block(in_ch=128, out_ch=256)

        # ────────────────────────────────────────────────────────────
        # STAGE 5: Global Average Pooling (GAP)
        # ────────────────────────────────────────────────────────────
        #
        # AdaptiveAvgPool2d((1, 1)) computes the MEAN of each channel's
        # feature map, collapsing spatial dimensions entirely:
        #
        #   [B, 256, H', W'] → [B, 256, 1, 1]
        #
        # MATH:
        #   GAP_c = (1 / H'W') · Σ_h Σ_w feature_map_c(h, w)
        #
        # WHY GAP instead of Flatten + Linear?
        #   1. Input-size agnostic: works on ANY spatial resolution
        #      (128×128, 224×224, etc.) because it always outputs [C, 1, 1].
        #   2. Far fewer parameters: Flatten would produce a 256×8×8 =
        #      16,384-dimensional vector → 16,384 × num_classes weights.
        #      GAP produces a 256-dimensional vector → 256 × num_classes.
        #   3. Reduces overfitting: fewer parameters = less capacity
        #      to memorise training data.
        #   4. Structural regularisation: each channel's mean activation
        #      serves as a natural "importance score" for the feature
        #      that channel detects.
        # ────────────────────────────────────────────────────────────

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ────────────────────────────────────────────────────────────
        # STAGE 6: Dropout
        # ────────────────────────────────────────────────────────────
        #
        # During training, randomly zeroes out elements with probability
        # p (default 0.5), and scales the remaining values by 1/(1-p)
        # so the expected sum stays the same (inverted dropout).
        #
        # MATH:
        #   mask_i ~ Bernoulli(1 - p)
        #   output_i = input_i · mask_i / (1 - p)
        #
        # WHY?
        #   Prevents co-adaptation: neurons can't rely on specific
        #   other neurons always being present, forcing the network to
        #   learn redundant representations.
        #
        # During inference (model.eval()), dropout is disabled —
        # all neurons participate.
        # ────────────────────────────────────────────────────────────

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # ────────────────────────────────────────────────────────────
        # STAGE 7: Fully Connected Classification Head
        # ────────────────────────────────────────────────────────────
        #
        # A single linear layer that maps the 256-dimensional feature
        # vector to num_classes raw logits (un-normalised scores).
        #
        # MATH:
        #   logits = W · x + b
        #
        #   where W is [num_classes × 256], b is [num_classes].
        #
        # WHY raw logits and NOT softmax?
        #   PyTorch's CrossEntropyLoss INCLUDES LogSoftmax internally.
        #   Applying softmax here and then passing to CrossEntropyLoss
        #   would double-apply softmax, producing incorrect gradients.
        #
        #   Softmax is only applied in predict() for human-readable
        #   probability output.
        # ────────────────────────────────────────────────────────────

        self.fc = nn.Linear(in_features=256, out_features=self.num_classes)

    # ═══════════════════════════════════════════════════════════════
    # Helper: Build a single convolutional block
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _build_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Construct one Conv → BatchNorm → ReLU → MaxPool block.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels (= number of 3×3 kernels).

        Returns
        -------
        nn.Sequential
            The assembled block. Output spatial size = input / 2.
        """
        return nn.Sequential(
            # ── Convolution ──────────────────────────────────────
            # kernel_size=3: each filter is 3×3 pixels
            # padding=1: add one row/col of zeros on each edge
            #   → output H, W = input H, W (before pooling)
            # bias=False: BatchNorm has its own bias (β)
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),

            # ── Batch Normalisation ──────────────────────────────
            # Normalise activations per channel across the batch
            nn.BatchNorm2d(out_ch),

            # ── Activation ───────────────────────────────────────
            # inplace=True: modifies the tensor in-place to save
            # memory (no need to keep the pre-ReLU values around,
            # PyTorch's autograd handles this via saved tensors)
            nn.ReLU(inplace=True),

            # ── Spatial Downsampling ─────────────────────────────
            # Takes max of each 2×2 window: [H, W] → [H/2, W/2]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    # ═══════════════════════════════════════════════════════════════
    # Forensic Channel Augmentation
    # ═══════════════════════════════════════════════════════════════

    def _add_forensic_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment a batch of RGB images with the configured forensic
        feature channels (FFT, LBP, Sobel).

        This method is called INSIDE forward() before the channel
        adapter. It loops over the batch dimension because each
        forensic function (compute_fft, compute_lbp, compute_sobel)
        operates on a single image [C, H, W], not on a batch.

        Parameters
        ----------
        x : torch.Tensor
            Batch of RGB images, shape [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Shape [B, 3+K, H, W] where K = number of enabled channels.
        """
        # If no forensic channels are enabled, return the input unchanged
        if not (self.add_fft or self.add_lbp or self.add_sobel):
            return x

        augmented_batch = []

        for i in range(x.shape[0]):
            # Extract single image: [3, H, W]
            img = x[i]

            # Collect channels to concatenate
            channels = [img]  # Start with original RGB

            if self.add_fft:
                channels.append(compute_fft(img))      # [1, H, W]
            if self.add_lbp:
                channels.append(compute_lbp(img))      # [1, H, W]
            if self.add_sobel:
                channels.append(compute_sobel(img))    # [1, H, W]

            # Concatenate along channel dim: [3+K, H, W]
            augmented = torch.cat(channels, dim=0)
            augmented_batch.append(augmented)

        # Stack back into batch: [B, 3+K, H, W]
        return torch.stack(augmented_batch, dim=0)

    # ═══════════════════════════════════════════════════════════════
    # Forward Pass
    # ═══════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of the FTL CNN.

        DATA FLOW
        ----------
        1. Add forensic channels (if configured):
             [B, 3, H, W]  →  [B, 3+K, H, W]

        2. Channel adapter (1×1 conv or Identity):
             [B, 3+K, H, W]  →  [B, 3, H, W]

        3. Conv blocks 1–4:
             [B, 3, H, W]  →  [B, 256, H/16, W/16]

        4. Global Average Pooling:
             [B, 256, H/16, W/16]  →  [B, 256, 1, 1]

        5. Flatten + Dropout:
             [B, 256, 1, 1]  →  [B, 256]

        6. FC head:
             [B, 256]  →  [B, num_classes]  (raw logits)

        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape [B, 3, H, W], values in [0, 1].

        Returns
        -------
        torch.Tensor
            Raw logits, shape [B, num_classes]. NOT probabilities —
            apply softmax externally if needed.
        """
        # Step 1: Augment with forensic channels
        logger.info(f"[CNN] forward pass started. Input shape: {x.shape}")
        x = self._add_forensic_channels(x)

        # Step 2: Mix channels down to 3 (or pass through if no extras)
        x = self.feature_adapter(x)
        logger.info(f"[CNN] Post feature-adapter shape: {x.shape}")

        # Step 3: Convolutional feature extraction
        x = self.block1(x)   # [B, 32,  H/2,  W/2]
        x = self.block2(x)   # [B, 64,  H/4,  W/4]
        x = self.block3(x)   # [B, 128, H/8,  W/8]
        x = self.block4(x)   # [B, 256, H/16, W/16]

        # Step 4: Global Average Pooling → [B, 256, 1, 1]
        x = self.global_avg_pool(x)

        # Step 5: Flatten to [B, 256] and apply dropout
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        # Step 6: Classification head → [B, num_classes]
        logits = self.fc(x)
        logger.info(f"[CNN] forward pass ended. Logits shape: {logits.shape}")

        return logits

    # ═══════════════════════════════════════════════════════════════
    # Transfer Learning Helpers
    # ═══════════════════════════════════════════════════════════════

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freezes or unfreezes the lower convolutional layers for transfer learning / fine-tuning.
        When frozen, only Conv Block 4 and the Classification Head are trained.
        """
        for module in [self.feature_adapter, self.block1, self.block2, self.block3]:
            for param in module.parameters():
                param.requires_grad = not freeze
        logger.info(f"[{self.__class__.__name__}] Transfer Learning: Backbone {'frozen' if freeze else 'unfrozen'}.")

    # ═══════════════════════════════════════════════════════════════
    # Single-Image Inference
    # ═══════════════════════════════════════════════════════════════

    @track_hardware
    @torch.no_grad()
    def predict(self, image_path: str) -> DetectionResult:
        """
        Classify a single image and return a structured DetectionResult.

        This method:
          1. Adds a batch dimension:    [C, H, W] → [1, C, H, W]
          2. Runs the forward pass:     → [1, num_classes] logits
          3. Applies softmax:           → [1, num_classes] probabilities
          4. Extracts the prediction:   label, confidence, prob dict
          5. Constructs a DetectionResult dataclass

        The @torch.no_grad() decorator disables gradient computation
        for efficiency — we don't need gradients for inference, and
        skipping them saves memory and compute.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image, shape [C, H, W], values in [0, 1].

        Returns
        -------
        DetectionResult
            Structured result with label, confidence, probabilities,
            and metadata.
        """
        # Import here to avoid circular dependency at module load time.
        # The TYPE_CHECKING guard at the top handles IDE autocompletion,
        # but the actual import must happen at call time.
        from core.results.detection import DetectionResult

        # Switch to evaluation mode (disables dropout + uses running
        # BatchNorm statistics instead of per-batch statistics)
        self.eval()

        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)

        # Move image to the model's device
        image_tensor = image_tensor.to(device)

        # Add batch dimension: [C, H, W] → [1, C, H, W]
        if image_tensor.dim() == 3:
            batch = image_tensor.unsqueeze(0)
        else:
            batch = image_tensor

        # Forward pass → raw logits [1, num_classes]
        logits = self.forward(batch)

        # Apply softmax to get probabilities
        #
        # SOFTMAX MATH:
        #   P(class_i) = e^(logit_i) / Σ_j e^(logit_j)
        #
        # This converts raw logits (which can be any real number) into
        # a proper probability distribution that sums to 1.
        # dim=1 means "normalise across classes for each sample".
        probs = F.softmax(logits, dim=1)

        # Extract results from the probability vector
        # probs shape: [1, num_classes] → squeeze to [num_classes]
        probs_squeezed = probs.squeeze()

        # torch.argmax returns the index of the highest probability
        predicted_idx = torch.argmax(probs_squeezed).item()

        # The confidence is the probability of the predicted class
        confidence = probs_squeezed[predicted_idx].item()

        # Build the human-readable probability dictionary
        # e.g. {"Real": 0.15, "AI-Generated": 0.85}
        probabilities = {
            CLASS_NAMES[i]: probs_squeezed[i].item()
            for i in range(self.num_classes)
        }

        # Is it a deepfake? Class 1 = AI-Generated
        is_deepfake = predicted_idx == 1

        # Construct and return the DetectionResult
        return DetectionResult(
            ai_deepfake=is_deepfake,
            confidence=confidence,
            image=image_path,
            model=self,
            returned_obj=probabilities,
        )

    # ═══════════════════════════════════════════════════════════════
    # Training Loop
    # ═══════════════════════════════════════════════════════════════

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        freeze_backbone: bool = False,
    ) -> dict[str, list[float]]:
        # Move model to device
        self.to(device)

        if freeze_backbone:
            self.freeze_backbone(True)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Loss function: see docstring above for the math
        criterion = nn.CrossEntropyLoss()

        # History dictionary to track metrics across epochs
        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
        }
        if val_loader is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        for epoch in range(epochs):
            # ── Training Phase ───────────────────────────────────
            self.train()  # Enable dropout + batch norm training mode

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                # Move data to device (GPU if available)
                images = images.to(device)
                labels = labels.to(device)

                # (a) Clear gradients from the previous iteration.
                #     PyTorch ACCUMULATES gradients by default (useful
                #     for gradient accumulation, but we don't want that
                #     in a standard loop).
                optimizer.zero_grad()

                # (b) Forward pass: builds the computation graph
                #     model(images) calls self.forward(images) internally,
                #     but also runs any registered hooks (needed for
                #     Grad-CAM later).
                logits = self(images)

                # (c) Compute loss: how wrong are the predictions?
                loss = criterion(logits, labels)

                # (d) Backpropagation: compute ∂loss/∂weight for every
                #     trainable parameter via the chain rule, walking
                #     backward through the computation graph.
                loss.backward()

                # (e) Update weights: θ_new = θ_old - lr · gradient
                #     (Adam does this with momentum + adaptive rates)
                optimizer.step()

                # ── Track metrics ────────────────────────────────
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # Compute epoch-level metrics
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            # ── Validation Phase (optional) ──────────────────────
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}"
                )

        return history

    # ═══════════════════════════════════════════════════════════════
    # Validation / Evaluation
    # ═══════════════════════════════════════════════════════════════

    @torch.inference_mode()
    def _evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """
        Evaluate the model on a dataset (validation or test).

        Uses @torch.inference_mode() to skip gradient computation — we only
        need the forward pass for evaluation.

        Parameters
        ----------
        data_loader : DataLoader
            Yields (images, labels) batches.
        criterion : nn.Module
            Loss function (same one used in training).

        Returns
        -------
        tuple[float, float]
            (average_loss, accuracy)
        """
        self.eval()  # Disable dropout + use running BN stats

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = self(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    # ═══════════════════════════════════════════════════════════════
    # XAI Hook: Target Layer for Grad-CAM
    # ═══════════════════════════════════════════════════════════════

    def get_target_layer(self) -> Optional[nn.Module]:
        """
        Return the last convolutional block for Grad-CAM.

        Grad-CAM needs to hook into a convolutional layer to capture
        feature map activations and their gradients. The LAST conv
        block (block4) is the best choice because:

          1. It has the largest receptive field — each spatial position
             in block4's feature map "sees" the most context from the
             original image.

          2. It has the most channels (256) — the richest feature
             representation, encoding high-level concepts like "face
             texture", "background consistency", etc.

          3. It's closest to the classifier head — its activations
             have the most direct influence on the final prediction.

        Returns
        -------
        nn.Sequential
            The 4th (last) convolutional block.
        """
        return self.block4

    # ═══════════════════════════════════════════════════════════════
    # Utility: Model Summary
    # ═══════════════════════════════════════════════════════════════

    def summary(self, input_size: tuple[int, int] = (128, 128)) -> str:
        """
        Print a human-readable summary of the model architecture.

        Shows each layer, its output shape, and parameter count.

        Parameters
        ----------
        input_size : tuple[int, int]
            (H, W) of the expected input images.

        Returns
        -------
        str
            Formatted string with the model summary.
        """
        h, w = input_size
        lines = [
            f"{'='*65}",
            f"  {self.name} — FTL CNN Detector",
            f"  Forensic channels: FFT={self.add_fft}, "
            f"LBP={self.add_lbp}, Sobel={self.add_sobel}",
            f"  Input channels: {self.input_channels} -> 3 (after adapter)",
            f"{'='*65}",
            f"  {'Layer':<30} {'Output Shape':<20} {'Params':>10}",
            f"  {'-'*60}",
        ]

        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()

        # Trace shapes through the network
        shapes = [
            ("Input", f"[B, {self.input_channels}, {h}, {w}]"),
            ("Feature Adapter (1×1)", f"[B, 3, {h}, {w}]"),
            ("Conv Block 1", f"[B, 32, {h//2}, {w//2}]"),
            ("Conv Block 2", f"[B, 64, {h//4}, {w//4}]"),
            ("Conv Block 3", f"[B, 128, {h//8}, {w//8}]"),
            ("Conv Block 4", f"[B, 256, {h//16}, {w//16}]"),
            ("Global Avg Pool", "[B, 256, 1, 1]"),
            ("Flatten + Dropout", "[B, 256]"),
            ("FC Head", f"[B, {self.num_classes}]"),
        ]

        for layer_name, shape in shapes:
            lines.append(f"  {layer_name:<30} {shape:<20}")

        lines.append(f"  {'-'*60}")
        lines.append(f"  Total parameters: {total_params:,}")
        lines.append(f"  Trainable parameters: "
                     f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        lines.append(f"{'='*65}")

        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str

    def get_target_layer(self) -> Optional[nn.Module]:
        """
        Return the last convolutional layer (Block 4 Conv2d) for Grad-CAM.
        """
        return self.block4[0]


# Backward-compatibility alias
FakeDetectorCNN = CNNDetector
