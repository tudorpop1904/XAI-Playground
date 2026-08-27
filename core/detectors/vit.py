"""
core/detectors/vit.py
======================
Vision Transformer (ViT) detector for deepfake image classification.

WHAT IS A VISION TRANSFORMER?
------------------------------
A CNN slides a small kernel across an image to extract local features
(edges, textures) and builds up to global understanding through
stacking conv layers. A Vision Transformer (ViT) does the opposite:
it cuts the image into fixed-size patches, treats each patch as a
"word" in a sequence, and lets every patch attend to every other
patch from the very first layer using self-attention.

This means:
  - A CNN has a limited receptive field per layer (3x3 kernel).
    It needs 4+ stacked layers before one position can "see" the
    whole image.
  - A ViT has GLOBAL receptive field from layer 1 -- every patch can
    attend to every other patch immediately.

For deepfake detection, this is particularly interesting because
AI-generated artifacts (inconsistent lighting, repeated textures,
frequency anomalies) often span distant regions of the image. A
ViT can detect these long-range inconsistencies directly.

ARCHITECTURE OVERVIEW
----------------------
    Input Image [3, H, W]
         |
         v
    Patch Embedding
    Split image into PxP patches,
    flatten each, project to D dims
    [3, H, W] -> [N, D]
    where N = (H/P) x (W/P)
         |
         v
    [CLS] Token Prepend
    [N, D] -> [N+1, D]
    Learnable classification token
         |
         v
    Positional Embedding
    Add learned position to each
    token so the model knows WHERE
    each patch came from
         |
         v
    L x Transformer Encoder Blocks
    Each block:
      LayerNorm -> MHSA -> Residual
      LayerNorm -> MLP  -> Residual
         |
         v
    Classification Head
    Extract [CLS] token -> LayerNorm
    -> Linear(D, num_classes)
         |
         v
    Raw logits [B, num_classes]

PAPER REFERENCE
----------------
- A. Dosovitskiy et al. (ICLR 2021).
  "An Image is Worth 16x16 Words: Transformers for Image Recognition
  at Scale"
- Paper: https://arxiv.org/pdf/2010.11929
- Repository: https://github.com/google-research/vision_transformer
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from core.utils.metrics import track_hardware
from torch import nn
from torch.utils.data import DataLoader

from .base import AbstractBaseDetector, logger
from core.results.detection import DetectionResult


# -- Device-agnostic setup ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -- Class label mapping ---------------------------------------------------
CLASS_NAMES = {0: "Real", 1: "AI-Generated"}


# =========================================================================
#  BUILDING BLOCKS
#  We build the ViT from scratch using elementary PyTorch layers.
#  Each class corresponds to one box in the architecture diagram.
# =========================================================================


class PatchEmbedding(nn.Module):
    """
    Split an image into fixed-size patches and project each into a
    D-dimensional embedding vector.

    WHAT THIS DOES (intuitively)
    ----------------------------
    Imagine cutting a photograph into a grid of small squares (e.g.,
    16x16 pixels each). Each square is a "patch". We then flatten each
    patch into a 1D vector and project it to a fixed embedding size D
    using a learned linear transformation.

    This is mathematically equivalent to a single Conv2d with:
      kernel_size = patch_size
      stride      = patch_size   (non-overlapping patches)
      out_channels = embed_dim

    WHY Conv2d INSTEAD OF MANUAL RESHAPE + LINEAR?
    Both approaches are mathematically identical, but Conv2d is:
      1. More memory-efficient (no intermediate flatten)
      2. Faster on GPU (optimised CUDA kernels for convolution)
      3. Exactly what the original ViT paper uses in practice

    MATH
    -----
    Given an input image x in R^{C x H x W}:

    1. Number of patches:  N = (H / P) x (W / P)
       where P = patch_size.

    2. Each patch is a vector of length C x P x P.

    3. The projection maps each patch to R^D:
         z_i = W . patch_i + b
       where W in R^{D x (C.P.P)} and b in R^D.

    Parameters
    ----------
    img_size : int
        Height/width of the square input image (e.g., 224).
    patch_size : int
        Height/width of each square patch (e.g., 16).
    in_channels : int
        Number of input channels (3 for RGB).
    embed_dim : int
        Dimension D of each patch embedding vector.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Number of patches along one axis
        self.grid_size = img_size // patch_size

        # Total number of patches (N)
        self.num_patches = self.grid_size ** 2

        # The projection: Conv2d with kernel=patch_size, stride=patch_size
        # Input:  [B, C, H, W]
        # Output: [B, D, H/P, W/P]  ->  reshape to [B, N, D]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Shape [B, N, D] where N = num_patches.
        """
        # Conv2d: [B, C, H, W] -> [B, D, grid_size, grid_size]
        x = self.proj(x)

        # Flatten spatial dims and transpose:
        # [B, D, grid_size, grid_size] -> [B, D, N] -> [B, N, D]
        x = torch.flatten(x, start_dim=2).permute(0, 2, 1)

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) -- the core mechanism of the
    Transformer.

    WHAT THIS DOES (intuitively)
    ----------------------------
    Each patch asks: "Which other patches in this image should I pay
    attention to?" A patch showing a nose might attend strongly to
    patches showing eyes and mouth (they form a face). A patch of sky
    might attend to other sky patches to verify colour consistency.

    For deepfake detection, self-attention is powerful because a
    patch containing an AI artifact (e.g., a distorted ear) can
    attend to the symmetric counterpart (the other ear) and detect
    inconsistencies that a CNN would need many layers to notice.

    MATH
    -----
    Given input X in R^{N x D}:

    1. Project X into three representations per head h:
         Q_h = X . W_Q^h,   K_h = X . W_K^h,   V_h = X . W_V^h
       where W_Q, W_K, W_V in R^{D x d_k} and d_k = D / num_heads.

    2. Compute attention weights:
         A_h = softmax(Q_h . K_h^T / sqrt(d_k))

       The division by sqrt(d_k) prevents the dot products from growing
       too large (which would push softmax into saturation, killing
       gradients).

    3. Compute weighted sum of values:
         head_h = A_h . V_h

    4. Concatenate all heads and project back:
         MHSA(X) = Concat(head_1, ..., head_H) . W_O

    WHY MULTIPLE HEADS?
    Each head learns a different "type" of attention pattern:
      - Head 1 might learn spatial proximity (attend to neighbours)
      - Head 2 might learn colour consistency (attend to similar hues)
      - Head 3 might learn structural symmetry (attend to mirror patches)

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension D.
    num_heads : int
        Number of attention heads H. Must divide D evenly.
    attn_drop : float
        Dropout rate on the attention weights.
    proj_drop : float
        Dropout rate on the output projection.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_k
        self.scale = self.head_dim ** -0.5       # 1 / sqrt(d_k)

        # Single linear layer that computes Q, K, V simultaneously
        # Output dim = 3 * embed_dim (Q, K, V concatenated)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Output projection: concatenated heads -> embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # -- Store attention weights for XAI (attention rollout) ------
        # This tensor is populated during forward() and read by
        # get_attention_map() for explainability.
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, N+1, D] (N patches + 1 CLS token).

        Returns
        -------
        torch.Tensor
            Shape [B, N+1, D].
        """
        B, N, D = x.shape

        # Step 1: Compute Q, K, V in one shot
        # [B, N, D] -> [B, N, 3*D] -> [B, N, 3, H, d_k] -> [3, B, H, N, d_k]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = torch.unbind(qkv, dim=0)  # Each: [B, H, N, d_k]

        # Step 2: Scaled dot-product attention
        # attn = softmax(Q . K^T / sqrt(d_k))
        # [B, H, N, d_k] @ [B, H, d_k, N] -> [B, H, N, N]
        attn = (q @ k.transpose(-2,-1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        # Store for XAI -- detach to avoid leaking gradients into the
        # attention rollout computation
        self.attn_weights = attn.detach()

        attn = self.attn_drop(attn)

        # Step 3: Weighted sum of values
        # [B, H, N, N] . [B, H, N, d_k] -> [B, H, N, d_k]
        x = torch.tensordot(attn, v, dims=3)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, N, D)

        # Step 4: Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer Encoder block.

    STRUCTURE
    ----------
    Each block applies two sub-layers with residual connections:

        x = x + MHSA(LayerNorm(x))      <-- attention sub-layer
        x = x + MLP(LayerNorm(x))        <-- feed-forward sub-layer

    WHY PRE-NORM (LayerNorm BEFORE attention)?
    The original Transformer (Vaswani 2017) used post-norm. The ViT
    paper uses pre-norm because it leads to more stable training:
    gradients flow through the residual path unmodified, and the
    normalisation prevents the attention scores from exploding.

    THE MLP (Feed-Forward Network)
    Each token is independently processed by a 2-layer MLP:
        Linear(D -> 4D) -> GELU -> Dropout -> Linear(4D -> D) -> Dropout

    The expansion factor (4x) gives the network a "wider thinking
    space" to compute non-linear features, before projecting back
    to the original dimension.

    WHY GELU INSTEAD OF RELU?
    GELU (Gaussian Error Linear Unit) is smoother than ReLU near zero.
    This matters for Transformers because attention scores can produce
    values very close to zero, and GELU's smooth curve helps gradients
    flow better in that regime.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension D.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        MLP hidden dimension = embed_dim x mlp_ratio.
    drop : float
        Dropout rate for MLP and attention output.
    attn_drop : float
        Dropout rate for attention weights.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        # Pre-norm before attention
        self.norm1 = nn.LayerNorm(embed_dim)

        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Pre-norm before MLP
        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP (feed-forward network)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, N+1, D].

        Returns
        -------
        torch.Tensor
            Shape [B, N+1, D].
        """
        # Attention sub-layer with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP sub-layer with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


# =========================================================================
#  THE VISION TRANSFORMER DETECTOR
# =========================================================================


class ViTDetector(AbstractBaseDetector):
    """
    Vision Transformer (ViT-Tiny) for deepfake detection.

    This detector implements a small ViT architecture from scratch
    (no pretrained weights -- trained end-to-end on our forensic data).
    It follows the same AbstractBaseDetector contract as the CNN,
    making it a drop-in replacement in the classifier registry.

    DIFFERENCES FROM THE CNN DETECTOR
    -----------------------------------
    1. No FTL channels -- the ViT processes raw RGB. The self-attention
       mechanism is expressive enough to learn frequency and texture
       patterns directly from pixels (at the cost of needing more data).

    2. No Grad-CAM -- the ViT has no convolutional layers, so Grad-CAM
       (which hooks into conv feature maps) doesn't apply. Instead,
       we provide ATTENTION ROLLOUT as a native explainability method.

    3. Input resolution -- ViT-Tiny expects 224x224 input. If the user
       configures a different image size, we interpolate at inference
       time.

    CONFIGURATION (ViT-Tiny)
    -------------------------
    This follows the ViT-Tiny configuration from the DeiT paper
    (Touvron et al., 2021):

        Patch size  : 16 x 16
        Embed dim   : 192
        Depth       : 12 transformer blocks
        Heads       : 3
        MLP ratio   : 4.0
        Parameters  : ~5.7M (vs ~470K for our FTL CNN)

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 2).
    img_size : int
        Expected input image size (default: 224).
    patch_size : int
        Patch size P (default: 16).
    embed_dim : int
        Embedding dimension D (default: 192).
    depth : int
        Number of Transformer blocks (default: 12).
    num_heads : int
        Number of attention heads per block (default: 3).
    mlp_ratio : float
        MLP expansion ratio (default: 4.0).
    drop_rate : float
        Dropout rate (default: 0.1).
    attn_drop_rate : float
        Attention dropout rate (default: 0.0).

    Example
    -------
    >>> model = ViTDetector(num_classes=2, img_size=224)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([1, 2])
    """

    # -- Class-level counter for auto-naming instances --------------------
    index = 1

    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__(name=f"ViT_{ViTDetector.index}")
        ViTDetector.index += 1

        # Store configuration for serialisation / logging
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        # ----------------------------------------------------------------
        # STAGE 0: Patch Embedding
        # ----------------------------------------------------------------
        #
        # Cuts the image into a grid of PxP patches and projects each
        # into a D-dimensional vector. See PatchEmbedding class above.
        # ----------------------------------------------------------------
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ----------------------------------------------------------------
        # STAGE 1: CLS Token
        # ----------------------------------------------------------------
        #
        # A learnable vector prepended to the patch sequence. After
        # passing through all Transformer blocks, the CLS token has
        # "attended to" every patch in the image and aggregated their
        # information. We use it as the image-level representation
        # for classification (instead of, say, averaging all patches).
        #
        # Shape: [1, 1, D] -- broadcast across the batch dimension.
        #
        # WHY a separate token instead of averaging patches?
        # The CLS token is free to learn an OPTIMAL aggregation
        # strategy via attention, rather than being forced into a
        # simple mean. It can learn to weight important patches more
        # heavily and ignore background patches.
        # ----------------------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ----------------------------------------------------------------
        # STAGE 2: Positional Embedding
        # ----------------------------------------------------------------
        #
        # Self-attention is permutation-invariant -- it doesn't know
        # the ORDER of the patches. Without positional embeddings,
        # shuffling all patches would give the exact same output.
        #
        # We add a learned positional embedding to each token:
        #   z_i = patch_embed_i + pos_embed_i
        #
        # The model learns that position 0 = top-left, position N-1
        # = bottom-right, and the attention patterns can use this
        # spatial information.
        #
        # Shape: [1, N+1, D]  (N patches + 1 CLS token)
        # ----------------------------------------------------------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # ----------------------------------------------------------------
        # STAGE 3: Transformer Encoder Blocks
        # ----------------------------------------------------------------
        #
        # A stack of `depth` identical Transformer blocks. Each block
        # refines the patch representations via self-attention and MLP.
        #
        # After all blocks, the CLS token contains a summary of the
        # entire image, weighted by learned attention patterns.
        # ----------------------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # ----------------------------------------------------------------
        # STAGE 4: Classification Head
        # ----------------------------------------------------------------
        #
        # Extract the CLS token -> LayerNorm -> Linear(D, num_classes)
        #
        # LayerNorm before the head stabilises the final representation
        # (same reasoning as pre-norm in the blocks).
        # ----------------------------------------------------------------
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # -- Initialise weights -------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise learnable parameters using best practices from the
        ViT paper.

        - Positional embeddings: truncated normal (std=0.02)
        - CLS token: truncated normal (std=0.02)
        - Linear layers: Xavier uniform (balances variance across layers)
        - LayerNorm: bias=0, weight=1 (the "identity" starting point)
        """
        # Positional embedding + CLS token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Walk all submodules
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # =====================================================================
    #  Forward Pass
    # =====================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the Vision Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [B, 3, H, W]. If H, W != img_size,
            the input is bilinearly interpolated to img_size x img_size.

        Returns
        -------
        torch.Tensor
            Raw logits of shape [B, num_classes].
        """
        B = x.shape[0]

        # -- Resize if necessary -----------------------------------------
        # The patch embedding assumes img_size x img_size input.
        # If the user trains at 128x128 but the ViT expects 224x224,
        # we upscale here. This is a design trade-off: interpolation
        # adds slight blurring, but it lets us use a standard patch
        # size (16) without fractional patches.
        if x.shape[2] != self.img_size or x.shape[3] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        # Step 1: Patch embedding [B, 3, 224, 224] -> [B, N, D]
        logger.info(f"[ViT] forward pass started. Input shape: {x.shape}")
        x = self.patch_embed(x)
        logger.info(f"[ViT] Patch embedding generated shape: {x.shape}")

        # Step 2: Prepend CLS token
        # Expand cls_token from [1, 1, D] -> [B, 1, D] (one per image)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Step 3: Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Step 4: Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 5: Extract CLS token, normalise, classify
        cls_output = x[:, 0]      # [B, D] -- the first token
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)  # [B, num_classes]
        logger.info(f"[ViT] forward pass ended. Logits shape: {logits.shape}")

        return logits

    # =====================================================================
    #  Inference (Single Image)
    # =====================================================================

    @track_hardware
    @torch.no_grad()
    def predict(self, image_path: str) -> DetectionResult:
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
        from core.results.detection import DetectionResult

        self.eval()
        self.to(device)

        image_tensor = torch.load(Path(image_path).resolve(), weights_only=False)
        # Add batch dimension: [C, H, W] -> [1, C, H, W]
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)

        # Forward pass -> raw logits [1, num_classes]
        logits = self(x)

        # Convert logits to probabilities via softmax
        probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        # Extract prediction
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        # Build probability dictionary
        probabilities = {
            CLASS_NAMES[i]: round(probs[i].item(), 4)
            for i in range(self.num_classes)
        }

        return DetectionResult(
            ai_deepfake=(class_idx == 1),
            confidence=confidence,
            image=image_path,
            model=self,
            returned_obj=probabilities,
        )

    # =====================================================================
    #  Training Loop
    # =====================================================================

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> dict[str, list[float]]:
        """
        Train the ViT on a dataset.

        Uses AdamW (Adam with decoupled weight decay) -- the standard
        optimiser for Transformers. Weight decay acts as L2
        regularisation, penalising large weights to prevent overfitting.

        Unlike the CNN (which uses plain Adam), Transformers benefit
        significantly from weight decay because:
          1. Self-attention has a huge parameter space (Q, K, V per head
             per layer), and weight decay constrains it.
          2. The positional embeddings can overfit to training positions
             without regularisation.

        Parameters
        ----------
        train_loader : DataLoader
            Training data yielding (images, labels) batches.
        val_loader : DataLoader | None
            Validation data (optional).
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate (default: 3e-4, standard for Transformers).
        weight_decay : float
            L2 regularisation strength (default: 0.01).

        Returns
        -------
        dict[str, list[float]]
            Training history with keys:
            "train_loss", "train_acc", "val_loss", "val_acc".
        """
        self.to(device)
        self.train()

        # AdamW -- Adam with decoupled weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        # CrossEntropyLoss -- combines LogSoftmax + NLLLoss
        criterion = nn.CrossEntropyLoss()

        # Cosine annealing scheduler -- smoothly decays the learning
        # rate from `lr` to near-zero over the training run.
        # This helps fine-tune in the later epochs.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # -- Track metrics ----------------------------------------
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            # Step the scheduler after each epoch
            scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            # -- Validation Phase (optional) ------------------------------
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        return history

    # =====================================================================
    #  Validation / Evaluation
    # =====================================================================

    @torch.inference_mode()
    def _evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """
        Evaluate the model on a dataset (validation or test).

        Parameters
        ----------
        data_loader : DataLoader
            Yields (images, labels) batches.
        criterion : nn.Module
            Loss function.

        Returns
        -------
        tuple[float, float]
            (average_loss, accuracy)
        """
        self.eval()

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

    # =====================================================================
    #  XAI: Attention Rollout
    # =====================================================================

    @torch.inference_mode()
    def get_attention_map(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute an attention rollout map for a single image.

        WHAT IS ATTENTION ROLLOUT?
        --------------------------
        Each Transformer block produces an attention matrix A_l of shape
        [H, N+1, N+1] that describes how much each token attends to
        every other token at layer l.

        Attention rollout multiplies these matrices across all layers to
        get a single matrix showing the CUMULATIVE attention from the
        CLS token to every patch:

            R = A_1 . A_2 . ... . A_L

        We then extract the CLS token's row (row 0) from R, discard the
        CLS-to-CLS entry, and reshape the remaining N values into a
        spatial grid. This gives us a heatmap showing which patches the
        model paid the most attention to.

        MATH
        -----
        At each layer l, the raw attention matrix is A_l in R^{(N+1) x (N+1)}.

        To account for residual connections (which mix the original
        representation with the attention output), we add the identity
        matrix and re-normalise:

            A_bar_l = 0.5 * A_l + 0.5 * I

        This models the fact that each token keeps ~50% of its own
        information and mixes in ~50% from attention.

        Then we accumulate:
            R_l = A_bar_l . R_{l-1},  with R_0 = I

        The final heatmap is the CLS token's row of R_L.

        WHY THIS IS BETTER THAN GRAD-CAM FOR ViTs
        -------------------------------------------
        Grad-CAM hooks into convolutional feature maps. A ViT has no
        convolutional feature maps (except the patch embedding, which
        is too early to be meaningful). Attention rollout is the native
        explainability method for Transformers -- it directly shows the
        model's learned attention patterns.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image tensor of shape [C, H, W], values in [0, 1].

        Returns
        -------
        torch.Tensor
            Attention heatmap of shape [grid_size, grid_size],
            values in [0, 1].
        """
        self.eval()
        self.to(device)

        # Forward pass to populate attn_weights in each block
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)
        _ = self(x)

        # Collect attention matrices from all blocks
        # Each attn_weights has shape [B, H, N+1, N+1]
        # Average across heads -> [N+1, N+1]
        result = None

        for block in self.blocks:
            attn = block.attn.attn_weights  # [1, H, N+1, N+1]
            attn = attn.squeeze(0).mean(dim=0)  # [N+1, N+1]

            # Account for residual connection: A_bar = 0.5*A + 0.5*I
            attn = 0.5 * attn + 0.5 * torch.eye(
                attn.size(0), device=attn.device
            )

            # Re-normalise rows to sum to 1
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Accumulate: R_l = A_bar_l . R_{l-1}
            if result is None:
                result = attn
            else:
                result = attn @ result

        # Extract CLS token's attention to all patches
        # result[0, :] = CLS token's attention weights
        # result[0, 0] = CLS-to-CLS (skip it)
        # result[0, 1:] = CLS-to-patches
        cls_attention = result[0, 1:]  # [N]

        # Reshape to spatial grid
        grid_size = self.patch_embed.grid_size
        attn_map = cls_attention.reshape(grid_size, grid_size)

        # Normalise to [0, 1]
        attn_map = attn_map - attn_map.min()
        if attn_map.max() > 1e-8:
            attn_map = attn_map / attn_map.max()

        return attn_map.cpu()

    # =====================================================================
    #  XAI Hook: Target Layer for Grad-CAM
    # =====================================================================

    def get_target_layer(self) -> Optional[nn.Module]:
        """
        Return the target layer for gradient-based XAI.

        ViTs don't have traditional conv layers for Grad-CAM. However,
        the patch embedding projection IS a Conv2d, so we return it
        as a fallback. For proper ViT explainability, use
        get_attention_map() instead.

        Returns
        -------
        nn.Conv2d
            The patch embedding projection layer.
        """
        return self.patch_embed.proj

    # =====================================================================
    #  Utility: Model Summary
    # =====================================================================

    def summary(self, input_size: tuple[int, int] = (224, 224)) -> str:
        """
        Print a human-readable summary of the ViT architecture.

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
        N = self.patch_embed.num_patches
        D = self.embed_dim

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        lines = [
            f"{'='*65}",
            f"  {self.name} -- Vision Transformer (ViT-Tiny)",
            f"  Patch size: {self.patch_size}x{self.patch_size}, "
            f"Embed dim: {D}, Depth: {self.depth}, Heads: {self.num_heads}",
            f"{'='*65}",
            f"  {'Layer':<30} {'Output Shape':<20}",
            f"  {'-'*55}",
            f"  {'Input':<30} [B, 3, {h}, {w}]",
            f"  {'Patch Embedding':<30} [B, {N}, {D}]",
            f"  {'+ CLS Token':<30} [B, {N+1}, {D}]",
            f"  {'+ Positional Embedding':<30} [B, {N+1}, {D}]",
            f"  {'Transformer x{}'.format(self.depth):<30} [B, {N+1}, {D}]",
            f"  {'  -- MHSA ({} heads)'.format(self.num_heads):<30} [B, {N+1}, {D}]",
            f"  {'  -- MLP (ratio 4.0)':<30} [B, {N+1}, {D}]",
            f"  {'Extract CLS + LayerNorm':<30} [B, {D}]",
            f"  {'Classification Head':<30} [B, {self.num_classes}]",
            f"  {'-'*55}",
            f"  Total parameters: {total_params:,}",
            f"  Trainable parameters: {trainable:,}",
            f"{'='*65}",
        ]

        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str
