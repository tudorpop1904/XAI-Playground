# `image_features.py` — Deep Dive Explanation

## Overview

This file implements **forensic feature engineering** — the process of extracting additional signal channels from an image that a CNN can use to distinguish real photographs from AI-generated fakes.

The standard input to a CNN is a 3-channel **RGB** image. The Faster-Than-Lies (FTL) methodology augments this with **three forensic channels**, producing a **6-channel** input:

```
Input:  [R, G, B, FFT, LBP, Sobel]  →  shape [6, H, W]
```

Each forensic channel captures a different property of the image that AI generators tend to get wrong.

---

## Architecture: Where This File Fits

```
Image (PIL/file)
     │
     ▼
 torchvision.transforms  →  [3, H, W] RGB tensor
     │
     ▼
 add_forensic_channels()  ←── THIS FILE
     │
     ▼
 [6, H, W] augmented tensor
     │
     ▼
 1×1 Conv (in CNN)  →  [3, H, W] "super-channels"
     │
     ▼
 Conv blocks → GAP → FC → prediction
```

The key insight is that these three channels provide **domain-specific priors** to the CNN. Without them, the CNN would need to *learn* to detect frequency artifacts, texture anomalies, and edge inconsistencies purely from raw pixels. With them, we give the CNN a head start — the forensic information is pre-computed and handed to it on a plate.

---

## Helper Functions

### `_to_grayscale(image_tensor)`

Converts an RGB image to grayscale using the **ITU-R BT.601 luma formula**:

$$Y = 0.2989 \cdot R + 0.5870 \cdot G + 0.1140 \cdot B$$

> **Why not just average the channels?**  
> The human eye is not equally sensitive to all colours. We perceive green as the brightest, red as medium, and blue as the dimmest. The luma formula accounts for this, producing a grayscale image that matches *perceived* brightness rather than raw energy. This is the same formula OpenCV uses internally.

**Implementation detail:** The weights are stored in a tensor of shape `[3, 1, 1]`. When multiplied with the image tensor `[3, H, W]`, PyTorch's broadcasting rules automatically apply each weight to the corresponding channel across all pixels. The result is summed over the channel dimension with `keepdim=True` to produce `[1, H, W]`.

### `_min_max_normalize(tensor)`

Rescales any tensor to the `[0, 1]` range:

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \varepsilon}$$

The epsilon (`1e-8`) prevents division by zero when all values are identical (e.g., a solid-colour image where `x_max = x_min`).

---

## Channel 1: FFT (Fast Fourier Transform)

### The Intuition

Every image can be thought of as a sum of waves. Smooth gradients (sky, blurred backgrounds) are **low-frequency** waves. Sharp edges and fine texture (hair, fabric weave) are **high-frequency** waves.

The FFT decomposes the image into its constituent frequencies, telling us *how much* of each frequency is present.

### Why It Detects AI Images

AI generators (especially GANs) use **transposed convolutions** (also called "deconvolutions") to upsample low-resolution feature maps into high-resolution images. When the upsampling stride doesn't evenly divide the kernel size, the transposed convolution produces a **checkerboard pattern** — a periodic artifact that is invisible to the naked eye in the spatial domain but shows up as bright dots at specific frequencies in the FFT spectrum.

```
Spatial Domain (looks fine):    Frequency Domain (artifact visible):
┌──────────────────────┐        ┌──────────────────────┐
│                      │        │          ·           │
│   Generated Face     │   FFT  │      ·   ★   ·      │
│                      │  ────▶ │          ·           │
│                      │        │     Bright dots =    │
│                      │        │     periodic artifact│
└──────────────────────┘        └──────────────────────┘
```

### The Math, Step by Step

#### Step 1: Grayscale

FFT operates on scalar fields. Colour is irrelevant for frequency analysis.

#### Step 2: 2D Discrete Fourier Transform

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

Where:
- $f(x, y)$ = pixel intensity at position $(x, y)$
- $F(u, v)$ = complex coefficient at frequency $(u, v)$
- $M, N$ = image dimensions
- $j = \sqrt{-1}$ (imaginary unit)

Each output value $F(u, v)$ is a **complex number** encoding:
- **Magnitude** $|F(u,v)|$ = how strong the wave at frequency $(u,v)$ is
- **Phase** $\angle F(u,v)$ = where the wave is positioned (we discard this)

In code: `torch.fft.fft2(gray)` — computes this for the entire image at once.

#### Step 3: Shift Zero-Frequency to Centre

By default, the FFT output places the DC (zero-frequency) component at corner `[0, 0]`. We rearrange it so:

```
Before fftshift:          After fftshift:
┌──────┬──────┐          ┌──────┬──────┐
│ DC   │ high │          │ high │ high │
│      │ freq │          │ freq │ freq │
├──────┼──────┤   ────▶  ├──────┼──────┤
│ high │ high │          │ high │  DC  │
│ freq │ freq │          │ freq │      │
└──────┴──────┘          └──────┴──────┘
```

Now the centre = low frequencies, edges = high frequencies. This makes the spectrum visually interpretable.

In code: `torch.fft.fftshift(fft, dim=(-2, -1))`

#### Step 4: Magnitude

$$|F(u,v)| = \sqrt{\text{Re}(F)^2 + \text{Im}(F)^2}$$

In code: `torch.abs(fft_shifted)` — when applied to a complex tensor, PyTorch computes the above.

#### Step 5: Log Compression

$$M_{\log} = \log(1 + |F(u,v)|)$$

The DC component can be $10^6 \times$ larger than high-frequency components. Without log compression, the visualisation would be a bright dot in the centre and black everywhere else. `log1p` (i.e. $\log(1+x)$) compresses the dynamic range and avoids $\log(0) = -\infty$.

#### Step 6: Normalise to [0, 1]

Standard min-max normalisation so the channel matches the RGB value range.

### Code Summary

```python
gray = _to_grayscale(image_tensor)     # [3,H,W] → [1,H,W]
fft = torch.fft.fft2(gray)             # Complex [1,H,W]
fft_shifted = torch.fft.fftshift(fft)  # Centre DC
magnitude = torch.abs(fft_shifted)     # Real [1,H,W]
magnitude = torch.log1p(magnitude)     # Compress range
magnitude = _min_max_normalize(magnitude)  # → [0,1]
```

---

## Channel 2: LBP (Local Binary Pattern)

### The Intuition

LBP answers the question: **"What does the texture look like at this pixel?"**

For each pixel, we look at its 8 neighbours and ask a simple yes/no question: "Is this neighbour brighter than the centre?" The 8 answers (bits) form a binary number that encodes the local texture pattern.

### Why It Detects AI Images

Real photographs have **rich, diverse micro-textures** — skin pores, fabric weave, tree bark, concrete grain. Each small region has a distinctive LBP pattern.

AI generators struggle with micro-texture because:
1. They operate at a higher level of abstraction (learning "face", "hair", "skin" as concepts rather than individual pores)
2. The generator's latent space doesn't have enough capacity to encode every unique texture detail
3. As a result, AI images have **unnaturally uniform** LBP distributions — the same texture patterns repeat too often

### The Math, Step by Step

#### The 3×3 Neighbourhood

```
┌────────┬────────┬────────┐
│  p₇    │  p₆    │  p₅    │
│(-1,-1) │(-1, 0) │(-1,+1) │
├────────┼────────┼────────┤
│  p₀    │ centre │  p₄    │
│( 0,-1) │( 0, 0) │( 0,+1) │
├────────┼────────┼────────┤
│  p₁    │  p₂    │  p₃    │
│(+1,-1) │(+1, 0) │(+1,+1) │
└────────┴────────┴────────┘
```

#### Comparison and Binary Encoding

For each neighbour $p_k$:

$$\text{bit}_k = \begin{cases} 1 & \text{if } p_k \geq \text{centre} \\ 0 & \text{if } p_k < \text{centre} \end{cases}$$

The LBP value at the centre pixel is:

$$\text{LBP} = \sum_{k=0}^{7} \text{bit}_k \cdot 2^k$$

This produces a value in $[0, 255]$ (an 8-bit integer).

#### Worked Example

```
Pixel values:              Comparison with centre (120):
┌─────┬─────┬─────┐       ┌───┬───┬───┐
│ 135 │ 100 │ 125 │       │ 1 │ 0 │ 1 │   (bit 7, 6, 5)
├─────┼─────┼─────┤       ├───┼───┼───┤
│ 130 │ 120 │  90 │       │ 1 │ - │ 0 │   (bit 0, -, 4)
├─────┼─────┼─────┤       ├───┼───┼───┤
│ 115 │ 140 │ 110 │       │ 0 │ 1 │ 0 │   (bit 1, 2, 3)
└─────┴─────┴─────┘       └───┴───┴───┘

Reading bits in order (p0→p7): 1, 0, 1, 0, 0, 1, 0, 1

LBP = 1·1 + 0·2 + 1·4 + 0·8 + 0·16 + 1·32 + 0·64 + 1·128
    = 1 + 4 + 32 + 128
    = 165
```

### Implementation: Vectorised with `torch.roll`

Instead of looping over every pixel (which would be extremely slow in Python), we use **shifted copies** of the entire image:

```python
# Shift the image so pixel (y,x) now contains the value of neighbour (y+dy, x+dx)
neighbour = torch.roll(gray, shifts=(-dy, -dx), dims=(-2, -1))

# Compare: is neighbour >= centre?
bit = (neighbour >= gray).float()  # → 0.0 or 1.0

# Accumulate with bit weight
lbp = lbp + bit * (2 ** k)
```

This processes **all pixels simultaneously** — a key advantage of working in pure PyTorch rather than looping in Python.

> **Edge wrapping:** `torch.roll` wraps pixels around the edges (the top row's "above" neighbour is the bottom row). This introduces a 1-pixel border artifact, but since the CNN has its own convolution padding, this is negligible in practice.

### Final Normalisation

Since the maximum LBP value is 255 (all 8 bits set), we simply divide by 255:

$$\text{LBP}_{\text{norm}} = \frac{\text{LBP}}{255}$$

---

## Channel 3: Sobel (Edge Magnitude)

### The Intuition

The Sobel operator is a **discrete approximation of the image gradient**. It answers: **"How rapidly is the intensity changing at this pixel, and in which direction?"**

Edges appear wherever there's a sharp transition in brightness — object boundaries, shadow edges, texture boundaries.

### Why It Detects AI Images

AI generators produce edge artifacts because:
1. **Over-sharp edges:** The generator "overcommits" to rendering object boundaries, producing edges sharper than physics would allow
2. **Missing edges:** In complex regions (hair, foliage, fabric folds), the generator blurs away fine edge detail
3. **Inconsistent edges:** Depth-of-field inconsistencies — edges are sharp where they should be blurred (or vice versa)

### The Math, Step by Step

#### The Two Sobel Kernels

The Sobel operator uses two $3 \times 3$ convolution kernels:

**Horizontal gradient** $G_x$ (detects vertical edges):

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}$$

**Vertical gradient** $G_y$ (detects horizontal edges):

$$G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

#### Why These Specific Values?

Each kernel is the **outer product of a smoothing vector and a derivative vector**:

$$G_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \cdot \begin{bmatrix} -1 & 0 & 1 \end{bmatrix}$$

- The column $[1, 2, 1]^T$ is a discrete **Gaussian smoothing** along the y-axis (reduces noise)
- The row $[-1, 0, 1]$ is a **central difference** along the x-axis (computes the derivative)

So the Sobel operator **smooths in one direction while differentiating in the other**, making it more noise-resistant than a simple finite difference.

#### Edge Magnitude

The gradient at each pixel is a 2D vector $(G_x, G_y)$. Its magnitude:

$$M = \sqrt{G_x^2 + G_y^2}$$

gives the **edge strength** regardless of direction.

#### Worked Example

Consider this $5 \times 5$ image patch with a vertical edge:

```
Image:                    Gx convolution at centre pixel:
┌───┬───┬───┬───┬───┐
│ 10│ 10│ 10│100│100│    Gx = (-1)(10) + (0)(10) + (1)(100)
├───┼───┼───┼───┼───┤         +(-2)(10) + (0)(10) + (2)(100)
│ 10│ 10│ 10│100│100│         +(-1)(10) + (0)(10) + (1)(100)
├───┼───┼───┼───┼───┤       = -10 + 100 - 20 + 200 - 10 + 100
│ 10│ 10│ 10│100│100│       = 360
├───┼───┼───┼───┼───┤
│ 10│ 10│ 10│100│100│    Gy ≈ 0 (no vertical intensity change)
├───┼───┼───┼───┼───┤
│ 10│ 10│ 10│100│100│    M = sqrt(360² + 0²) = 360 → strong edge!
└───┴───┴───┴───┴───┘
```

### Implementation with `F.conv2d`

```python
# Define kernels as [1, 1, 3, 3] tensors (batch=1, channels=1, 3x3)
sobel_x = tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
sobel_y = tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)

# Convolve (padding=1 for same-size output)
gx = F.conv2d(gray_4d, sobel_x, padding=1)
gy = F.conv2d(gray_4d, sobel_y, padding=1)

# Edge magnitude
magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
```

The epsilon (`1e-8`) inside the sqrt prevents gradient issues during backpropagation: the derivative of $\sqrt{x}$ at $x = 0$ is undefined, but $\sqrt{x + \varepsilon}$ is always well-defined.

---

## The Stacking Function: `add_forensic_channels()`

This is the function the CNN calls at the start of every forward pass:

```python
def add_forensic_channels(image_tensor):
    fft   = compute_fft(image_tensor)    # [1, H, W]
    lbp   = compute_lbp(image_tensor)    # [1, H, W]
    sobel = compute_sobel(image_tensor)  # [1, H, W]
    
    # [3,H,W] + [1,H,W] + [1,H,W] + [1,H,W] = [6,H,W]
    return torch.cat([image_tensor, fft, lbp, sobel], dim=0)
```

The output `[6, H, W]` is then fed into a **1×1 convolution** (`nn.Conv2d(6, 3, kernel_size=1)`) which learns to mix the 6 channels into 3 "super-channels":

$$\text{super\_channel}_i = \sum_{j=1}^{6} w_{ij} \cdot \text{channel}_j + b_i \quad \text{for } i \in \{1, 2, 3\}$$

This is mathematically equivalent to a **per-pixel fully-connected layer** across the channel dimension — it doesn't look at spatial neighbours, only at the 6 channel values at each pixel position.

---

## Design Decision: Why Pure PyTorch?

The original code (in `xai-app/`) used `numpy`, `cv2`, and `skimage` for these computations. The new implementation uses **pure PyTorch** for three critical reasons:

| Aspect | NumPy/CV2 approach | Pure PyTorch approach |
|--------|-------------------|----------------------|
| **GPU support** | ❌ CPU only | ✅ Runs on GPU if available |
| **Gradient flow** | ❌ Breaks autograd | ✅ Gradients flow through |
| **Data copies** | ❌ CPU↔GPU copies | ✅ Zero-copy, same device |
| **Batch processing** | ❌ Loop over images | ✅ Naturally batched |

The gradient flow point is especially important: if we want Grad-CAM or saliency maps to "see through" the forensic channels (i.e., understand *which forensic features* contributed to the prediction), the feature computation must be part of the differentiable computation graph. With numpy, the graph would be severed.

---

## Summary Table

| Channel | What It Captures | AI Artifact It Reveals | Function |
|---------|-----------------|----------------------|----------|
| **FFT** | Frequency content | Checkerboard patterns from transposed convolutions | `compute_fft()` |
| **LBP** | Local texture structure | Unnaturally uniform micro-textures | `compute_lbp()` |
| **Sobel** | Edge magnitude | Over-sharp or missing edges | `compute_sobel()` |

Each channel transforms the image into a **different representation space** where AI artifacts become visible, even when they're invisible in the raw RGB pixel space.
