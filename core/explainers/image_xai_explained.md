# Image XAI — Mathematics Explained

This document serves as the mathematical companion for the XAI methods implemented in `core/xai/explainers/`. Explainable AI (XAI) attempts to answer the question: *"Why did the model classify this image as X?"*

We implement two categories of methods: **White-box** (requires access to model architecture and gradients) and **Black-box** (model-agnostic, perturbation-based).

---

## 1. White-Box Methods (Gradient-Based)

These methods use backpropagation to trace the prediction score back to the pixels or internal feature maps.

### 1.1 Vanilla Saliency
*Paper: Simonyan et al. (2013) "Deep Inside Convolutional Networks..."*

**The Intuition:** 
If we change a specific pixel by a tiny amount, how much does the probability of the "AI-Generated" class change? Pixels that cause the biggest change are the most important.

**The Math:**
Given an image `X` and the model's logit output for the target class `Y_c`:

1. Compute the derivative of the target logit with respect to the input image:
   ```
   W = ∂Y_c / ∂X    (shape: [C, H, W])
   ```
2. Because a pixel might have a strong negative OR positive influence, we take the absolute value. Since there are 3 color channels, we take the maximum across channels to get a single 2D map:
   ```
   SaliencyMap(i, j) = max_{c ∈ {R,G,B}} |W_{c, i, j}|
   ```

**Pros:** Extremely fast (1 backward pass). Works on any differentiable model.
**Cons:** Shattered gradients — it often highlights high-frequency noise and edges rather than semantically meaningful object parts.

### 1.2 Grad-CAM
*Paper: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations..."*

**The Intuition:**
Instead of looking at raw pixels, we look at the **last convolutional layer's** feature maps. These maps detect high-level semantic features (like "distorted eye" or "smooth skin"). Grad-CAM figures out which of these feature maps are most responsible for the prediction, and weights them accordingly.

**The Math:**
Let `A^k` be the k-th feature map in the final conv layer. Let `Y_c` be the logit for the target class.

1. Compute the gradient of the logit with respect to the feature maps:
   ```
   G^k = ∂Y_c / ∂A^k
   ```
2. Global Average Pool (GAP) the gradients to get a single weight `α^k` for each feature map. This weight represents how important feature map `k` is for class `c`:
   ```
   α^k = (1 / Z) * Σ_i Σ_j (G^k_{i, j})
   ```
3. Compute the weighted sum of the forward feature maps, and apply a ReLU (we only care about features that have a *positive* influence on the class, not those that suppress it):
   ```
   L_{Grad-CAM} = ReLU( Σ_k α^k A^k )
   ```
4. Resize this small heatmap (e.g., 7x7) to the original image dimensions (224x224).

**Pros:** Class-discriminative, highlights semantic regions smoothly.
**Cons:** Resolution is tied to the last conv layer (usually very coarse, 7x7). Only works on CNNs.

---

## 2. Black-Box Methods (Perturbation-Based)

These methods treat the model as a black box: `f(x) = p`. They divide the image into a grid of cells, hide/reveal cells, and measure how `p` changes.

Let `P_base` be the probability of the target class on the original image `X`.
Let `Mask_{r,c}` be an operation that occludes cell `(r, c)`.

### 2.1 Occlusion Sensitivity
*Paper: Zeiler & Fergus (2014) "Visualizing and Understanding Convolutional Networks"*

**The Intuition:**
Slide a grey box over the image. If the probability drops heavily when a specific area is hidden, that area was important.

**The Math:**
For each cell `(r, c)`:
```
P_occ = f( X ⊙ Mask_{r,c} )
Score(r, c) = max(0, P_base - P_occ)
```
*(We use `max(0, ...)` because if occluding a cell INCREASES the probability, it means that cell was actually evidence AGAINST the target class, so its score for the target class should be 0).*

**Pros:** Very intuitive, easy to implement.
**Cons:** Highly sensitive to the size of the grey box.

### 2.2 Visual Pointwise Mutual Information (PMI)
*Inspired by NLP PMI masking techniques.*

**The Intuition:**
Instead of hiding one cell, we hide **everything except** one cell, and see how much information that single cell provides compared to a completely grey image.

**The Math:**
Let `P_grey` be the probability when the entire image is grey.
Let `P_reveal` be the probability when ONLY cell `(r, c)` is visible.

```
PMI(r, c) = log_2( P_reveal / P_grey )
Score(r, c) = max(0, PMI(r, c))
```

**Pros:** Excellent for detecting isolated deepfake artifacts (e.g., a single warped tooth that gives away the fake on its own).
**Cons:** Destroys global context; models might act unpredictably when seeing 90% grey images.

### 2.3 Visual Sobol (Monte Carlo Variance Reduction)
*Paper: Petsiuk et al. (2018) "RISE: Randomized Input Sampling for Explanation"*

**The Intuition:**
Instead of hiding one cell at a time, we generate N random binary masks (e.g., 50% of the image is randomly occluded in chunks). We pass all N masked images through the model and get N probabilities. 

For a specific cell, we compare the variance of ALL N predictions against the variance of the subset of predictions where that specific cell was VISIBLE. If fixing that cell to be visible massively reduces the variance of the model's output, it means the model heavily relies on that cell to make a confident decision.

**The Math:**
1. Generate N random binary masks `M_1, ..., M_N`.
2. Compute `p_i = f(X ⊙ M_i)` for all `i`.
3. Compute total variance: `V_total = Var({p_1, ..., p_N})`
4. For each cell `(r, c)`:
   - Find subset of predictions where the cell was visible: `S_{r,c} = {p_i | M_i(r,c) == 1}`
   - Compute conditional variance: `V_cond = Var(S_{r,c})`
   - Compute Sobol sensitivity index:
   ```
   Score(r, c) = max(0, (V_total - V_cond) / V_total)
   ```

**Pros:** Extremely robust, less sensitive to mask size than single-occlusion. Captures interactions between different regions.
**Cons:** Computationally expensive (requires N forward passes, where N is usually 64 to 1000).

---

## 3. Computational Complexity Summary

| Method | Type | Forward Passes | Backward Passes | Output Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Saliency** | White-box | 1 | 1 | High (Pixel-level) |
| **Grad-CAM** | White-box | 1 | 1 | Low (e.g., 7x7) |
| **Occlusion** | Black-box | `R × C + 1` | 0 | Grid (e.g., 4x4) |
| **Visual PMI** | Black-box | `R × C + 2` | 0 | Grid (e.g., 4x4) |
| **Sobol** | Black-box | `N + 1` | 0 | Grid (e.g., 4x4) |

*Where `R` and `C` are the number of grid rows and columns, and `N` is the number of Monte Carlo samples.*
