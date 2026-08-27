"""
core/xai/image_features.py
===========================
Forensic Feature Engineering for AI-Generated Image Detection
-------------------------------------------------------------

This module implements the three supplementary forensic channels described
in the "Faster-Than-Lies" (FTL) deepfake detection methodology:

    1. FFT  — Fast Fourier Transform magnitude spectrum
    2. LBP  — Local Binary Pattern texture descriptor
    3. Sobel — Edge magnitude via Sobel gradient operators

MOTIVATION (Why these channels?)
---------------------------------
A standard CNN receives only 3 channels: R, G, B. These channels capture
colour and spatial intensity, but they do NOT explicitly encode:

    - Frequency content  → FFT reveals periodic artifacts left by GANs
                           (e.g., checkerboard patterns from transposed
                           convolutions in the generator's upsampling)
    - Micro-texture       → LBP captures the local neighbourhood structure
                           that AI generators often homogenise
    - Edge sharpness      → Sobel highlights unnaturally sharp or blurred
                           boundaries that GANs produce at object edges

By stacking these three 1-channel maps alongside the original RGB, we get
a 6-channel input tensor [B, 6, H, W]. A 1×1 convolution layer in the CNN
then learns to mix the 6 channels into 3 "super-channels" before entering
the standard conv blocks.

DESIGN DECISIONS
-----------------
• Pure PyTorch — all operations use torch.* so they run on GPU if available
  (no numpy/cv2 conversion needed, no CPU←→GPU copies during forward pass)
• Each function accepts [C, H, W] and returns [1, H, W]
  (ready to torch.cat along dim=0 with the original image)
• Min-max normalisation maps each channel to [0, 1] to match the RGB range

PAPER REFERENCE
----------------
• R. Lanzino, F. Fontana, A. Diko, M. R. Marini, L. Cinque (CVPRW 2024).
  "Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks"
• Paper: https://iris.uniroma1.it/retrieve/59badfa8-a0da-4a27-b85c-fad48d771c23/Lanzino_postprint_Faster_2024.pdf
• Repository: https://github.com/fedeloper/binary_deepfake_detection

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device-agnostic setup
# ---------------------------------------------------------------------------
# We define the device once so every function can use it consistently.
# torch.cuda.is_available() returns True only if an NVIDIA GPU with CUDA
# drivers is present AND PyTorch was compiled with CUDA support.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: RGB → Grayscale
# ═══════════════════════════════════════════════════════════════════════════

def _to_grayscale(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image tensor to grayscale using the ITU-R BT.709 luma
    formula.

    MATH
    ----
    Y = 0.2126·R + 0.7152·G + 0.0722·B

    This is a *perceptual* weighting — the human eye is most sensitive
    to green, moderately to red, and least to blue. Using a simple mean
    (R+G+B)/3 would over-weight blue relative to perceived brightness.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape [C, H, W] with C=3 (RGB), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Shape [1, H, W], grayscale, values in [0, 1].
    """
    # Luma weights for R, G, B (ITU-R BT.709 standard)
    # Shape: [3, 1, 1] so broadcasting with [3, H, W] yields [1, H, W]
    weights = torch.tensor(
        [0.2126, 0.7152, 0.0722],
        dtype=image_tensor.dtype,
        device=image_tensor.device
    ).view(3, 1, 1)

    # Element-wise multiply [3,H,W] * [3,1,1] then sum over channel dim
    # Result shape: [1, H, W]
    gray = (image_tensor * weights).sum(dim=0, keepdim=True)

    return gray


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Min-Max Normalisation
# ═══════════════════════════════════════════════════════════════════════════

def _min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Linearly rescale tensor values to the range [0, 1].

    MATH
    ----
    x_norm = (x - x_min) / (x_max - x_min + ε)

    The epsilon (1e-8) prevents division by zero when the input is
    constant (all values the same), which can happen with a solid-
    colour image.

    Parameters
    ----------
    tensor : torch.Tensor
        Any shape. Normalisation is applied globally (across all elements).

    Returns
    -------
    torch.Tensor
        Same shape, values in [0, 1].
    """
    t_min = tensor.min()
    t_max = tensor.max()
    return (tensor - t_min) / (t_max - t_min + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# 1. FFT — Fast Fourier Transform Magnitude Spectrum
# ═══════════════════════════════════════════════════════════════════════════

def compute_fft(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-scaled FFT magnitude spectrum of an image.

    WHY FFT FOR DEEPFAKE DETECTION?
    --------------------------------
    Every image can be decomposed into a sum of sinusoidal waves at
    different frequencies and orientations. The FFT converts an image
    from the *spatial domain* (pixel values at x,y positions) to the
    *frequency domain* (amplitude of each frequency component).

    Real photographs have a characteristic 1/f spectral falloff (high
    frequencies are weaker than low frequencies). AI-generated images —
    especially those from GANs using transposed convolutions — often
    exhibit:
      • Periodic bright spots (checkerboard artifacts) at specific
        frequencies corresponding to the generator's upsampling stride
      • Spectral "gaps" or abnormally flat regions where the generator
        failed to reproduce the natural 1/f distribution

    ALGORITHM (step by step)
    -------------------------
    1. Convert RGB → grayscale
       (FFT operates on scalar fields; colour is irrelevant for
       frequency analysis)

    2. Apply 2D FFT: F(u,v) = Σ_x Σ_y f(x,y) · e^(-j2π(ux/M + vy/N))
       This produces a complex-valued matrix where:
         - Each entry F(u,v) encodes the amplitude and phase of the
           sinusoidal component at horizontal frequency u and vertical
           frequency v
         - The output has the same shape [H, W] as the input

    3. Shift zero-frequency to centre: fftshift()
       By default, the FFT places the zero-frequency (DC) component at
       the top-left corner. We shift it to the centre of the image so
       that:
         - Centre = low frequencies (smooth gradients, overall brightness)
         - Edges  = high frequencies (sharp details, noise, texture)
       This makes the spectrum visually interpretable.

    4. Compute magnitude: |F(u,v)| = sqrt(Re² + Im²)
       We discard the phase information and keep only the amplitude
       at each frequency. Phase encodes spatial position; magnitude
       encodes how *strong* each frequency is.

    5. Log compression: log(1 + |F(u,v)|)
       The raw magnitude spectrum has enormous dynamic range — the DC
       component can be 10⁶× larger than high-frequency components.
       The log compresses this range so both low and high frequencies
       are visible. We use log1p (= log(1+x)) to avoid log(0) = -∞.

    6. Min-max normalise to [0, 1]
       So the channel has the same value range as RGB.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape [C, H, W] with C=3 (RGB), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Shape [1, H, W], the normalised log-magnitude spectrum.
    """
    # Ensure computation happens on the correct device
    image_tensor = image_tensor.to(device)

    # Step 1: RGB → Grayscale [3,H,W] → [1,H,W]
    gray = _to_grayscale(image_tensor)

    # Step 2: 2D FFT — output is complex-valued, shape [1,H,W]
    # torch.fft.fft2 applies the FFT over the last two dimensions (H, W)
    fft = torch.fft.fft2(gray)

    # Step 3: Shift zero-frequency component to the centre
    # dim=(-2, -1) means "shift along the H and W axes"
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Step 4: Magnitude = |F(u,v)|
    # torch.abs on a complex tensor computes sqrt(real² + imag²)
    magnitude = torch.abs(fft_shifted)

    # Step 5: Log compression — log(1 + magnitude)
    # torch.log1p(x) = log(1 + x), numerically stable for small x
    magnitude = torch.log1p(magnitude)

    # Step 6: Normalise to [0, 1]
    magnitude = _min_max_normalize(magnitude)

    return magnitude


# ═══════════════════════════════════════════════════════════════════════════
# 2. LBP — Local Binary Pattern
# ═══════════════════════════════════════════════════════════════════════════

def compute_lbp(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute a Local Binary Pattern (LBP) texture map using pure PyTorch.

    WHY LBP FOR DEEPFAKE DETECTION?
    ---------------------------------
    LBP encodes the *local texture structure* around each pixel. For each
    pixel, we compare it to its 8 immediate neighbours. This produces a
    binary pattern that describes the micro-texture at that point.

    Real photographs have diverse, organically varying micro-textures (skin
    pores, fabric weave, tree bark). AI-generated images often exhibit:
      • Unnaturally uniform LBP distributions (texture "smoothness")
      • Repeated identical patterns in regions where generators copy
        learned texture patches
      • Missing or over-simplified high-frequency texture details

    ALGORITHM (step by step)
    -------------------------
    1. Convert RGB → grayscale

    2. Compare the centre pixel to each of its 8 neighbours:
       ┌────────┬────────┬────────┐
       │ p7     │ p6     │ p5     │
       │(-1,-1) │(-1, 0) │(-1,+1) │
       ├────────┼────────┼────────┤
       │ p0     │ centre │ p4     │
       │( 0,-1) │( 0, 0) │( 0,+1) │
       ├────────┼────────┼────────┤
       │ p1     │ p2     │ p3     │
       │(+1,-1) │(+1, 0) │(+1,+1) │
       └────────┴────────┴────────┘

       For each neighbour pₖ:
         bit_k = 1  if pₖ ≥ centre
         bit_k = 0  if pₖ < centre

    3. Combine the 8 bits into a single integer:
         LBP = Σ_{k=0}^{7}  bit_k · 2^k

       This yields a value in [0, 255] (8-bit number) for each pixel.

       Example:
         Neighbours relative to centre (value=120):
           [130, 115, 140, 90, 125, 100, 135, 110]
         Comparisons (≥ 120?):
           [  1,   0,   1,  0,   1,   0,   1,   0]
         Binary: 10101010 = 170 in decimal

    4. Convert the integer map to float and normalise to [0, 1]
       (divide by 255, since max possible LBP value is 255)

    IMPLEMENTATION NOTE
    --------------------
    We implement LBP as 8 shifted copies of the image, each compared
    against the original. This is equivalent to sliding a 3×3 window
    but is fully vectorised — no Python loops over pixels.

    The shifts are applied using torch.roll(), which wraps pixels around
    the edges. This introduces a 1-pixel border artifact, but since the
    CNN has its own padding, this is negligible.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape [C, H, W] with C=3 (RGB), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Shape [1, H, W], the normalised LBP texture map.
    """
    image_tensor = image_tensor.to(device)

    # Step 1: RGB → Grayscale [1, H, W]
    gray = _to_grayscale(image_tensor)

    # Step 2 & 3: Compare with 8 neighbours and accumulate binary pattern
    #
    # The 8 neighbours are defined as (row_shift, col_shift) pairs.
    # We traverse them counter-clockwise starting from the left neighbour,
    # assigning each position a power of 2 (bit weight).
    #
    # neighbour_offsets[k] = (dy, dx)  →  bit weight = 2^k
    neighbour_offsets = [
        ( 0, -1),  # p0: left            → bit 0 (weight 1)
        ( 1, -1),  # p1: bottom-left     → bit 1 (weight 2)
        ( 1,  0),  # p2: bottom          → bit 2 (weight 4)
        ( 1,  1),  # p3: bottom-right    → bit 3 (weight 8)
        ( 0,  1),  # p4: right           → bit 4 (weight 16)
        (-1,  1),  # p5: top-right       → bit 5 (weight 32)
        (-1,  0),  # p6: top             → bit 6 (weight 64)
        (-1, -1),  # p7: top-left        → bit 7 (weight 128)
    ]

    # Start with a zero tensor that will accumulate the LBP values
    lbp = torch.zeros_like(gray)

    for k, (dy, dx) in enumerate(neighbour_offsets):
        # Shift the image so that position (y, x) now contains the value
        # of neighbour (y+dy, x+dx).
        #
        # torch.roll shifts along the given dimension:
        #   shifts=-dy on dim=-2 means: move row content UP by dy
        #     → pixel at (y, x) gets the value from (y+dy, x)
        #   shifts=-dx on dim=-1 means: move column content LEFT by dx
        #     → pixel at (y, x) gets the value from (y, x+dx)
        neighbour = torch.roll(gray, shifts=(-dy, -dx), dims=(-2, -1))

        # Compare: is the neighbour ≥ the centre pixel?
        # .float() converts boolean True/False to 1.0/0.0
        bit = (neighbour >= gray).float()

        # Accumulate: multiply by 2^k (the bit weight) and add
        lbp = lbp + bit * (2 ** k)

    # Step 4: Normalise to [0, 1]
    # Maximum possible LBP value is 255 (all 8 bits set), so divide by 255
    lbp = lbp / 255.0

    return lbp


# ═══════════════════════════════════════════════════════════════════════════
# 3. Sobel — Edge Magnitude
# ═══════════════════════════════════════════════════════════════════════════

def compute_sobel(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute edge magnitude using Sobel operators in pure PyTorch.

    WHY SOBEL FOR DEEPFAKE DETECTION?
    -----------------------------------
    The Sobel operator detects edges — sharp transitions in pixel intensity.
    Edges form at object boundaries, shadows, and texture discontinuities.

    AI-generated images often have:
      • Unnaturally sharp edges at object boundaries (the generator
        "overcommits" to edge sharpness)
      • Blurred or missing edges in areas the generator finds hard to
        reconstruct (fine hair, fabric folds, background details)
      • Inconsistent edge patterns where real images would have smooth
        gradients (e.g., at depth-of-field boundaries)

    ALGORITHM (step by step)
    -------------------------
    1. Convert RGB → grayscale

    2. Convolve with two 3×3 Sobel kernels:

       Horizontal gradient (Gx):        Vertical gradient (Gy):
       ┌────┬────┬────┐                 ┌────┬────┬────┐
       │ -1 │  0 │ +1 │                 │ -1 │ -2 │ -1 │
       ├────┼────┼────┤                 ├────┼────┼────┤
       │ -2 │  0 │ +2 │                 │  0 │  0 │  0 │
       ├────┼────┼────┤                 ├────┼────┼────┤
       │ -1 │  0 │ +1 │                 │ +1 │ +2 │ +1 │
       └────┴────┴────┘                 └────┴────┴────┘

       Gx approximates the partial derivative ∂I/∂x (horizontal change).
       Gy approximates the partial derivative ∂I/∂y (vertical change).

       The weights [-1, 0, +1] compute a finite difference (derivative),
       while the weights [1, 2, 1] along the perpendicular axis apply a
       Gaussian-like smoothing to reduce noise sensitivity.

    3. Compute edge magnitude:
         M = sqrt(Gx² + Gy²)

       This is the Euclidean norm of the gradient vector (Gx, Gy) at
       each pixel, giving the "strength" of the edge regardless of
       its orientation.

    4. Min-max normalise to [0, 1]

    IMPLEMENTATION NOTE
    --------------------
    We implement the Sobel convolution using F.conv2d with fixed (non-
    learnable) kernels. The kernels are shaped [1, 1, 3, 3] for conv2d:
    (out_channels=1, in_channels=1, kH=3, kW=3). Padding=1 ensures the
    output has the same spatial dimensions as the input (same-padding).

    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape [C, H, W] with C=3 (RGB), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Shape [1, H, W], the normalised edge magnitude map.
    """
    image_tensor = image_tensor.to(device)

    # Step 1: RGB → Grayscale [1, H, W]
    gray = _to_grayscale(image_tensor)

    # Step 2: Define Sobel kernels as fixed (non-learnable) convolution weights
    #
    # Shape for F.conv2d: [out_channels, in_channels, kH, kW]
    # We have 1 input channel (gray) and 1 output channel per kernel.

    # Gx kernel — detects vertical edges (intensity changes along x-axis)
    sobel_x = torch.tensor(
        [[-1.0,  0.0,  1.0],
         [-2.0,  0.0,  2.0],
         [-1.0,  0.0,  1.0]],
        dtype=image_tensor.dtype,
        device=image_tensor.device
    ).view(1, 1, 3, 3)  # Reshape to [1, 1, 3, 3] for conv2d

    # Gy kernel — detects horizontal edges (intensity changes along y-axis)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=image_tensor.dtype,
        device=image_tensor.device
    ).view(1, 1, 3, 3)

    # Add a batch dimension: [1, H, W] → [1, 1, H, W]
    # F.conv2d expects 4D input: [batch, channels, height, width]
    gray_4d = gray.unsqueeze(0)

    # Convolve with each kernel (padding=1 for same-size output)
    gx = F.conv2d(gray_4d, sobel_x, padding=1)  # [1, 1, H, W]
    gy = F.conv2d(gray_4d, sobel_y, padding=1)   # [1, 1, H, W]

    # Step 3: Edge magnitude = sqrt(Gx² + Gy²)
    # The small epsilon prevents sqrt(0) gradient issues during backprop
    magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # Remove the batch dimension: [1, 1, H, W] → [1, H, W]
    magnitude = magnitude.squeeze(0)

    # Step 4: Normalise to [0, 1]
    magnitude = _min_max_normalize(magnitude)

    return magnitude


# ═══════════════════════════════════════════════════════════════════════════
# 4. Stack All Forensic Channels
# ═══════════════════════════════════════════════════════════════════════════

def add_forensic_channels(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Augment an RGB image with the three forensic feature channels.

    This is the entry point called by the FTL CNN's forward() method.
    It takes the standard 3-channel RGB image and produces a 6-channel
    tensor by concatenating:

        [R, G, B, FFT, LBP, Sobel]

    The 1×1 convolution layer at the start of the CNN then mixes these
    6 channels down to 3 "super-channels" before the standard conv blocks.

    PIPELINE DIAGRAM
    -----------------
                        ┌──── R ────┐
         RGB Image      │           │
        [3, H, W]  ────▶├──── G ────┤
                        │           │
                        ├──── B ────┤
                        │           │──▶  [6, H, W]  ──▶  1×1 Conv  ──▶  CNN
         Forensic  ────▶├──── FFT ──┤
         Channels       │           │
                        ├──── LBP ──┤
                        │           │
                        └──── Sobel ┘

    Parameters
    ----------
    image_tensor : torch.Tensor
        Shape [C, H, W] with C=3 (RGB), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Shape [6, H, W], the original RGB channels concatenated with
        the FFT, LBP, and Sobel forensic channels.
    """
    # Compute each forensic channel — each returns [1, H, W]
    fft   = compute_fft(image_tensor)    # Frequency spectrum
    lbp   = compute_lbp(image_tensor)    # Texture pattern
    sobel = compute_sobel(image_tensor)  # Edge magnitude

    # Concatenate along the channel dimension (dim=0)
    # [3,H,W] + [1,H,W] + [1,H,W] + [1,H,W] = [6,H,W]
    stacked = torch.cat([image_tensor, fft, lbp, sobel], dim=0)

    return stacked
