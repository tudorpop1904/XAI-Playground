# `cnn.py` — Deep Dive Explanation

## Overview

This file implements `CNNDetector`, the primary deepfake detection model from the Faster-Than-Lies (FTL) methodology. It is a convolutional neural network that classifies images as **Real** or **AI-Generated**.

What makes it special compared to a generic image classifier is the **forensic feature engineering**: before the standard convolution pipeline, the input image is optionally augmented with FFT, LBP, and Sobel channels computed by `image_features.py`, then mixed back down through a 1×1 convolution.

---

## Architecture Diagram

```
Input Image [B, 3, H, W]
      │
      ▼
┌─────────────────────────────────────────────┐
│  Forensic Augmentation (if enabled)         │
│  Append FFT, LBP, Sobel channels            │
│  [B, 3, H, W] → [B, 3+K, H, W]            │
└──────────────────┬──────────────────────────┘
                   │
      ┌────────────▼─────────────┐
      │  1×1 Channel Adapter     │
      │  [B, 3+K, H, W] → [B, 3]│
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  Conv Block 1            │
      │  Conv(3→32) + BN + ReLU  │
      │  + MaxPool(2)            │
      │  [B, 32, H/2, W/2]      │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  Conv Block 2            │
      │  Conv(32→64) + BN + ReLU │
      │  + MaxPool(2)            │
      │  [B, 64, H/4, W/4]      │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  Conv Block 3            │
      │  Conv(64→128)+ BN + ReLU │
      │  + MaxPool(2)            │
      │  [B, 128, H/8, W/8]     │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  Conv Block 4            │  ← Grad-CAM hooks here
      │  Conv(128→256)+ BN + ReLU│
      │  + MaxPool(2)            │
      │  [B, 256, H/16, W/16]   │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  Global Average Pooling  │
      │  [B, 256, 1, 1]         │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  Flatten + Dropout(0.5)  │
      │  [B, 256]               │
      └────────────┬─────────────┘
                   │
      ┌────────────▼─────────────┐
      │  FC Head                 │
      │  [B, 256] → [B, 2]      │
      │  (raw logits)            │
      └──────────────────────────┘
```

---

## Stage 0: The 1×1 Channel Adapter

### What It Does

When forensic channels are enabled, the input has more than 3 channels (up to 6: R, G, B, FFT, LBP, Sobel). The rest of the CNN expects exactly 3 channels. The 1×1 convolution bridges this gap by learning a **weighted combination** of all input channels.

### The Math

For each pixel at position $(h, w)$:

$$\text{super\_channel}_i(h, w) = \sum_{j=1}^{C_{\text{in}}} w_{ij} \cdot \text{channel}_j(h, w)$$

where $C_{\text{in}} = 3 + K$ (K = number of forensic channels), and $i \in \{1, 2, 3\}$.

This is equivalent to a **per-pixel matrix multiplication**:

$$\begin{bmatrix} s_1 \\ s_2 \\ s_3 \end{bmatrix} = \underbrace{\begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1,C_{\text{in}}} \\ w_{21} & w_{22} & \cdots & w_{2,C_{\text{in}}} \\ w_{31} & w_{32} & \cdots & w_{3,C_{\text{in}}} \end{bmatrix}}_{3 \times C_{\text{in}} \text{ weight matrix}} \begin{bmatrix} R \\ G \\ B \\ \text{FFT} \\ \text{LBP} \\ \text{Sobel} \end{bmatrix}$$

### Why It Works

The network learns **which forensic channels matter** and **how to combine them** with the raw RGB. After training, you can inspect the weight matrix to see, for example, that the model learned to weight FFT heavily for detecting GAN artifacts while relying on LBP for texture analysis.

### Why `bias=False`?

The next layer (BatchNorm) already has a learnable bias parameter $\beta$. If we added a bias here too, we'd have:

$$y = \gamma \cdot \frac{(Wx + b_{\text{conv}}) - \mu}{\sigma} + \beta$$

The $b_{\text{conv}}$ gets absorbed into the normalisation — it shifts $\mu$ but then gets subtracted out. The optimiser would waste capacity learning two biases that cancel each other. Setting `bias=False` eliminates this redundancy.

---

## Stage 1–4: Convolutional Blocks

Each block follows the same pattern: **Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)**.

### Conv2d — Feature Extraction

A 3×3 convolution kernel slides across the input, computing a weighted sum at each position:

$$\text{output}(h, w) = \sum_{c=1}^{C_{\text{in}}} \sum_{i=-1}^{1} \sum_{j=-1}^{1} \text{kernel}(c, i, j) \cdot \text{input}(c, h+i, w+j)$$

With `padding=1`, the output has the **same spatial dimensions** as the input (before pooling).

**What the blocks progressively learn:**

| Block | Channels | What It Detects | Analogous To |
|-------|----------|----------------|--------------|
| 1 | 3 → 32 | Edges, colour gradients, simple textures | "There's a sharp edge here" |
| 2 | 32 → 64 | Combinations of edges, corners, texture patterns | "There's a face-like shape" |
| 3 | 64 → 128 | Parts of objects, complex textures | "This looks like skin texture" |
| 4 | 128 → 256 | High-level semantic features | "This skin texture looks artificially uniform" |

### BatchNorm2d — Stabilising Training

Normalises each channel's activations to have mean=0, std=1 across the batch:

$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \varepsilon}}$$

Then applies a learnable affine transform:

$$y = \gamma \cdot \hat{x} + \beta$$

**Why this matters:**

- Without BatchNorm, the distribution of each layer's inputs shifts as earlier layers update their weights. This "internal covariate shift" forces later layers to constantly readapt, slowing training.
- BatchNorm fixes the distribution, allowing higher learning rates and faster convergence.
- The learnable $\gamma$ and $\beta$ ensure the network can still represent any distribution it needs (if $\gamma = \sigma$ and $\beta = \mu$, BatchNorm becomes an identity operation).

**Training vs. Inference:**
- During training: $\mu$ and $\sigma^2$ are computed fresh from each mini-batch.
- During inference (`model.eval()`): $\mu$ and $\sigma^2$ are **running averages** accumulated during training. This ensures deterministic output regardless of batch composition.

### ReLU — Non-Linearity

$$f(x) = \max(0, x)$$

Without non-linearity, stacking linear layers is pointless:

$$W_2 \cdot (W_1 \cdot x) = (W_2 \cdot W_1) \cdot x = W_{\text{combined}} \cdot x$$

Two linear layers collapse into one. ReLU breaks this:
- Positive values pass through unchanged (gradient = 1, no vanishing)
- Negative values are zeroed out (creating sparsity)

### MaxPool2d(2) — Spatial Downsampling

Takes the maximum value in each 2×2 window:

```
Input:              Output:
┌───┬───┬───┬───┐   ┌───┬───┐
│ 1 │ 3 │ 2 │ 4 │   │ 5 │ 4 │
├───┼───┼───┼───┤ → ├───┼───┤
│ 5 │ 2 │ 1 │ 3 │   │ 8 │ 7 │
├───┼───┼───┼───┤   └───┴───┘
│ 4 │ 8 │ 6 │ 7 │
├───┼───┼───┼───┤
│ 3 │ 1 │ 2 │ 5 │
└───┴───┴───┴───┘
```

Halves spatial dimensions: $H \times W \to \frac{H}{2} \times \frac{W}{2}$.

**Why max (not average)?**
Max pooling preserves the **strongest activation** — if a feature is detected anywhere in the 2×2 window, it survives the pooling. Average pooling would dilute strong signals with weak neighbours.

---

## Stage 5: Global Average Pooling (GAP)

$$\text{GAP}_c = \frac{1}{H' \times W'} \sum_{h=1}^{H'} \sum_{w=1}^{W'} \text{feature\_map}_c(h, w)$$

Collapses each feature map to a single number: [B, 256, H', W'] → [B, 256, 1, 1].

### Why GAP Instead of Flatten + Large FC?

| Approach | Vector Size (128×128 input) | FC Params (2 classes) |
|----------|---------------------------|----------------------|
| Flatten after block4 | 256 × 8 × 8 = 16,384 | 32,770 |
| GAP | 256 | 514 |

GAP reduces parameters by **~64×**, massively reducing overfitting risk. It also makes the model **input-size agnostic** — a 224×224 image produces [B, 256, 14, 14] after the conv blocks, but GAP still outputs [B, 256, 1, 1].

---

## Stage 6: Dropout

During training, each neuron is randomly "dropped" with probability $p$:

$$\text{output}_i = \begin{cases} 0 & \text{with probability } p \\ \frac{\text{input}_i}{1 - p} & \text{with probability } 1 - p \end{cases}$$

The scaling by $\frac{1}{1-p}$ is called **inverted dropout** — it ensures the expected value of the output equals the input, so the network doesn't need to be rescaled at inference time.

**Why p = 0.5?** This is the information-theoretic optimum for regularisation. Each neuron is present in only half of the training updates, forcing the network to learn redundant representations that are robust to any individual neuron failing.

**At inference time** (`model.eval()`), dropout is disabled — all neurons participate.

---

## Stage 7: Fully Connected Head

$$\text{logits} = W \cdot x + b$$

where $W \in \mathbb{R}^{2 \times 256}$ and $b \in \mathbb{R}^2$.

### Why Raw Logits, Not Softmax?

PyTorch's `CrossEntropyLoss` **internally applies LogSoftmax**:

$$\text{CrossEntropyLoss}(x, y) = -\log\left(\frac{e^{x_y}}{\sum_j e^{x_j}}\right)$$

If we applied softmax in `forward()` and then used `CrossEntropyLoss`, we'd compute:

$$-\log\left(\frac{e^{\text{softmax}(x)_y}}{\sum_j e^{\text{softmax}(x)_j}}\right)$$

That's softmax-of-softmax — wrong and numerically unstable. So `forward()` returns raw logits, and softmax is only applied in `predict()` for human-readable probabilities.

---

## The Training Loop

### Step-by-Step Breakdown

```python
for epoch in range(epochs):
    model.train()                          # 1. Enable dropout + BN training
    for images, labels in train_loader:
        optimizer.zero_grad()              # 2. Clear old gradients
        logits = model(images)             # 3. Forward pass
        loss = criterion(logits, labels)   # 4. Compute loss
        loss.backward()                    # 5. Backpropagation
        optimizer.step()                   # 6. Update weights
```

### Why `zero_grad()`?

PyTorch **accumulates** gradients by default. Without `zero_grad()`, the gradient from batch N would be **added to** the gradient from batch N-1. This is actually useful in some cases (gradient accumulation for effectively larger batch sizes), but in a standard loop, we want fresh gradients each step.

### Why `model(images)` and NOT `model.forward(images)`?

`model(images)` calls `__call__()`, which:
1. Runs all registered **forward hooks** (needed for Grad-CAM)
2. Calls `forward()`
3. Runs all registered **backward hooks**

Calling `forward()` directly would skip the hooks, breaking XAI methods.

### Adam Optimizer

Adam maintains two running averages per parameter:

- **First moment** $m_t$ (momentum): exponential moving average of gradients
  $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

- **Second moment** $v_t$ (adaptive rate): exponential moving average of squared gradients
  $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

Update rule (with bias correction):
$$\theta_t = \theta_{t-1} - \frac{\text{lr} \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

Parameters that have consistently large gradients get smaller learning rates (they've already learned a lot). Parameters with small, noisy gradients get larger learning rates (they need more adjustment).

---

## The Predict Method

```python
@torch.no_grad()
def predict(self, image_tensor):
    self.eval()                           # Disable dropout + use running BN
    batch = image_tensor.unsqueeze(0)     # [C,H,W] → [1,C,H,W]
    logits = self.forward(batch)          # [1, 2]
    probs = F.softmax(logits, dim=1)      # [1, 2] probabilities
    ...
```

### Softmax

Converts raw logits to a probability distribution:

$$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Example: if logits = [1.5, 3.2]:

$$P(\text{Real}) = \frac{e^{1.5}}{e^{1.5} + e^{3.2}} = \frac{4.48}{4.48 + 24.53} = 0.154$$

$$P(\text{AI}) = \frac{e^{3.2}}{e^{1.5} + e^{3.2}} = \frac{24.53}{4.48 + 24.53} = 0.846$$

### `@torch.no_grad()`

Disables gradient tracking entirely. Since we're only doing inference (no `backward()` call), we don't need the computation graph. This:
- Saves memory (no need to store intermediate tensors for backprop)
- Speeds up computation (no gradient bookkeeping)

---

## `get_target_layer()` — The XAI Hook

Returns `self.block4` (the last convolutional block) for Grad-CAM to hook into.

**Why the last block?**
- Largest **receptive field**: each spatial position "sees" the most context
- Most **channels** (256): richest feature representation
- Most **direct influence** on the classifier head

This is how the Service-layer separation works in practice:

```python
# In an explainer (core/xai/explainers/grad_cam.py):
target_layer = detector.get_target_layer()
# Register hooks on target_layer to capture activations + gradients
# Run forward + backward through the detector
# Compute weighted combination of feature maps → heatmap
```

The detector doesn't need to know anything about Grad-CAM. It just exposes the layer.

---

## Parameter Count

For the full FTL model (with FFT + LBP + Sobel):

| Component | Parameters |
|-----------|-----------|
| Feature Adapter (1×1 conv, 6→3) | 18 |
| Block 1 (Conv 3→32 + BN) | 928 |
| Block 2 (Conv 32→64 + BN) | 18,560 |
| Block 3 (Conv 64→128 + BN) | 73,984 |
| Block 4 (Conv 128→256 + BN) | 295,424 |
| FC Head (256→2) | 514 |
| **Total** | **~389K** |

This is **tiny** by modern standards (ResNet-18 has 11M, ViT-tiny has 5M). The small size means:
- Fast training (minutes, not hours)
- Fast inference (suitable for edge deployment on Raspberry Pi)
- Lower risk of overfitting on small datasets

---

## Key Design Decision: Why Not Use a Pre-built Architecture?

The whole point of the FTL methodology is to demonstrate that a **simple, from-scratch CNN** with **clever forensic feature engineering** can compete with much larger pre-trained models. The forensic channels (FFT, LBP, Sobel) give the small CNN domain-specific priors that a generic architecture would need millions more parameters to learn from raw pixels alone.
