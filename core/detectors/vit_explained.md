# Vision Transformer (ViT) — Mathematics Explained

This document is the mathematical companion to `core/detectors/vit.py`.
Read the code comments for intuition; read this for the formal math.

---

## 1. Patch Embedding

### The Problem
A Transformer processes **sequences** of vectors. An image is a 3D tensor
`[C, H, W]`, not a sequence. We need to convert it.

### The Solution
Cut the image into a grid of non-overlapping patches of size `P × P`,
flatten each patch, and linearly project it to dimension `D`.

### The Math

Given input image **x** ∈ ℝ^{C × H × W}:

1. **Number of patches:**
   ```
   N = (H / P) × (W / P)
   ```
   For H = W = 224, P = 16:  N = 14 × 14 = **196 patches**.

2. **Each patch** is a flattened vector of length `C × P × P`:
   ```
   patch_i ∈ ℝ^{C·P·P} = ℝ^{3·16·16} = ℝ^768
   ```

3. **Linear projection** to embedding dimension D = 192:
   ```
   z_i = W_embed · patch_i + b_embed
   ```
   where `W_embed ∈ ℝ^{D × (C·P²)}` and `b_embed ∈ ℝ^D`.

4. **Result:** A sequence of N vectors, each of dimension D:
   ```
   Z = [z_1, z_2, ..., z_N] ∈ ℝ^{N × D}
   ```

### Implementation Trick
This is mathematically identical to `Conv2d(3, 192, kernel_size=16, stride=16)`.
The convolution slides a 16×16 kernel across the image with stride 16
(no overlap), producing a `[B, 192, 14, 14]` feature map. Reshaping
to `[B, 196, 192]` gives the same result as manual flatten + linear.

---

## 2. CLS Token & Positional Embedding

### CLS Token
A learnable vector `cls ∈ ℝ^D` is prepended to the sequence:
```
Z' = [cls, z_1, z_2, ..., z_N] ∈ ℝ^{(N+1) × D}
```
After all Transformer blocks, `cls` has attended to every patch and
contains an image-level summary. We use it (instead of averaging all
patches) because it can learn a non-uniform, optimised aggregation
via attention.

### Positional Embedding
Self-attention is **permutation-invariant**: shuffling the patches
would produce the exact same output. To inject spatial information:
```
Z'' = Z' + E_pos
```
where `E_pos ∈ ℝ^{(N+1) × D}` is a learned parameter matrix.

Each row of `E_pos` encodes the spatial position of that token.
After training, nearby positions tend to have similar embeddings
(the model learns a smooth spatial prior).

---

## 3. Multi-Head Self-Attention (MHSA)

This is the core mechanism that replaces convolution.

### Single-Head Attention

Given input `X ∈ ℝ^{(N+1) × D}`:

1. **Compute Query, Key, Value:**
   ```
   Q = X · W_Q,    K = X · W_K,    V = X · W_V
   ```
   where `W_Q, W_K, W_V ∈ ℝ^{D × D}`.

2. **Attention scores** (how much each token "looks at" every other):
   ```
   A = softmax(Q · K^T / √D)
   ```
   - `Q · K^T ∈ ℝ^{(N+1) × (N+1)}` — each entry is the dot product
     between a query and a key, measuring "relevance".
   - **Scaling by `√D`**: Without this, the dot products grow with D,
     pushing softmax into regions where gradients vanish.
     The variance of a dot product of two D-dimensional vectors is D,
     so dividing by √D normalises the variance to 1.
   - **Softmax**: Converts raw scores to a probability distribution
     (each row sums to 1).

3. **Weighted sum of values:**
   ```
   Attention(Q, K, V) = A · V ∈ ℝ^{(N+1) × D}
   ```

### Multi-Head Extension

Instead of one set of Q, K, V, we use H heads with smaller dimensions:
```
d_k = D / H    (for ViT-Tiny: 192 / 3 = 64)
```

Each head h computes:
```
Q_h = X · W_Q^h,   K_h = X · W_K^h,   V_h = X · W_V^h
head_h = softmax(Q_h · K_h^T / √d_k) · V_h
```

The heads are concatenated and projected:
```
MHSA(X) = Concat(head_1, ..., head_H) · W_O
```

### Why Multiple Heads?
Each head can specialise in a different pattern:
- **Head 1**: Attend to spatially adjacent patches (local texture)
- **Head 2**: Attend to patches with similar colour (consistency)
- **Head 3**: Attend to symmetric patches (structural verification)

For deepfake detection: a patch containing a distorted left ear can
attend to the right ear to detect asymmetry — something a 3×3 CNN
kernel physically cannot do in a single layer.

---

## 4. Transformer Block

Each of the L blocks applies:
```
x = x + MHSA(LayerNorm(x))     ← attention sub-layer
x = x + MLP(LayerNorm(x))       ← feed-forward sub-layer
```

### LayerNorm (Pre-Norm)
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```
where μ, σ² are the mean and variance across the D dimension of each
token, and γ, β are learnable affine parameters.

**Pre-norm** (before attention) is used instead of post-norm (after)
because it provides more stable gradients through the residual path.

### MLP (Feed-Forward Network)
```
MLP(x) = Linear_2(GELU(Linear_1(x)))
Linear_1: D → 4D    (expansion)
Linear_2: 4D → D    (projection back)
```

**GELU** (Gaussian Error Linear Unit):
```
GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```
where Φ is the standard Gaussian CDF. Unlike ReLU (which has a hard
zero at x < 0), GELU is smooth everywhere, helping gradient flow in
the near-zero regime where attention scores often land.

### Residual Connections
```
x_out = x_in + sublayer(x_in)
```
These ensure that gradients can flow directly from the output back to
the input without being attenuated by the sublayer. Even if the
sublayer's gradients vanish, the gradient through the skip connection
remains 1.

---

## 5. Classification

After L transformer blocks:
```
cls_output = LayerNorm(x[:, 0])     ← extract CLS token
logits = W_head · cls_output + b_head
```
where `W_head ∈ ℝ^{num_classes × D}`.

---

## 6. Attention Rollout (XAI)

### The Problem
Grad-CAM hooks into convolutional feature maps. ViTs have no conv
feature maps (except the patch embedding, which is too early). We
need a native ViT explainability method.

### The Solution: Attention Rollout

At each layer l, self-attention produces `A_l ∈ ℝ^{(N+1) × (N+1)}`.

**Step 1: Account for residual connections.**
The residual path means each token keeps ~50% of itself:
```
Ā_l = 0.5 · A_l + 0.5 · I
```
Re-normalise each row to sum to 1:
```
Ā_l = Ā_l / Ā_l.sum(dim=-1)
```

**Step 2: Accumulate across layers.**
```
R_0 = I
R_l = Ā_l · R_{l-1}
```

**Step 3: Extract CLS row.**
```
heatmap = R_L[0, 1:]     ← CLS token's attention to all N patches
```
Reshape to `(H/P, W/P)` = `(14, 14)` for ViT-Tiny with P=16.

### Interpretation
- **Bright regions**: The CLS token (and therefore the classifier)
  paid the most cumulative attention to these patches.
- **Dark regions**: These patches were largely ignored.
- For deepfake detection: if the model flags an image as AI-generated,
  the rollout map shows WHICH patches contained the most suspicious
  artifacts.

---

## 7. ViT vs CNN: Key Differences

| Property | CNN (FTL) | ViT-Tiny |
|---|---|---|
| **Basic operation** | 3×3 convolution (local) | Self-attention (global) |
| **Receptive field** | Grows with depth | Global from layer 1 |
| **Inductive bias** | Translation equivariance | None (more data-hungry) |
| **Parameters** | ~470K | ~5.7M |
| **FTL channels** | Yes (FFT, LBP, Sobel) | No (learns from RGB) |
| **XAI method** | Grad-CAM (hooks into conv) | Attention rollout (native) |
| **Optimiser** | Adam | AdamW (needs weight decay) |
| **LR schedule** | Flat | Cosine annealing |
| **Input size** | Flexible (GAP adapts) | Fixed (must interpolate) |

### When ViT > CNN
- Large, diverse datasets (more data compensates for lack of inductive bias)
- Long-range artifacts (frequency anomalies spanning the whole image)
- Multi-scale inconsistencies (lighting from one corner vs another)

### When CNN > ViT
- Small datasets (CNN's inductive bias helps it learn with less data)
- Fine-grained local texture (LBP/Sobel channels detect micro-patterns)
- Computational efficiency (12× fewer parameters, faster inference)

---

## 8. Parameter Count Breakdown (ViT-Tiny)

| Component | Shape | Parameters |
|---|---|---|
| Patch Embedding | Conv2d(3, 192, 16, 16) | 3 × 16 × 16 × 192 + 192 = **147,648** |
| CLS Token | [1, 1, 192] | **192** |
| Positional Embedding | [1, 197, 192] | **37,824** |
| Per Transformer Block: | | |
| ├─ QKV Linear | Linear(192, 576) | 192 × 576 + 576 = **111,168** |
| ├─ Output Proj | Linear(192, 192) | 192 × 192 + 192 = **37,056** |
| ├─ MLP Linear 1 | Linear(192, 768) | 192 × 768 + 768 = **148,224** |
| ├─ MLP Linear 2 | Linear(768, 192) | 768 × 192 + 192 = **147,648** |
| ├─ LayerNorm ×2 | 2 × (192 + 192) | **768** |
| └─ **Subtotal per block** | | **444,864** |
| × 12 blocks | | **5,338,368** |
| Final LayerNorm | 192 + 192 | **384** |
| Classification Head | Linear(192, 2) | 192 × 2 + 2 = **386** |
| **TOTAL** | | **~5,524,802** |
