# k-Nearest Neighbours (k-NN) — Mathematics Explained

This document is the mathematical companion to `core/detectors/knn.py`.

---

## 1. The Fundamental Idea

k-NN is the simplest classifier in machine learning. The algorithm is:

```
Given: a query image x, a training set {(x_1, y_1), ..., (x_N, y_N)}

1. Compute distance d(x, x_i) for every training sample i
2. Find the k samples with the smallest distances
3. The predicted class = majority vote among those k labels
```

No weights are learned. No gradients are computed. The model IS the
training data.

---

## 2. The Curse of Dimensionality (Why Raw Pixels Don't Work)

### The Problem
An image at 224×224×3 = **150,528 dimensions**. In very high-dimensional
spaces, a counterintuitive phenomenon occurs: **all points become
approximately equidistant**.

### The Math
For N points uniformly distributed in a D-dimensional hypercube [0, 1]^D,
the expected ratio between the nearest and farthest neighbour distances
converges to 1 as D → ∞:

```
lim_{D→∞}  E[d_max - d_min] / E[d_min]  →  0
```

This means that in 150K-dimensional pixel space, the "nearest" training
image is barely closer than the "farthest" — k-NN becomes meaningless.

### The Solution: Feature Extraction
Instead of comparing raw pixels, we first compress each image into a
**512-dimensional feature vector** using a pretrained ResNet-18 backbone.
These 512 dimensions encode meaningful semantic features (texture quality,
edge consistency, object structure) rather than raw pixel values.

```
Image [3, 224, 224]  →  ResNet-18  →  Feature [512]
   (150,528 dims)         (frozen)       (512 dims)
```

At 512 dimensions, the curse of dimensionality is manageable, and
distances between features are semantically meaningful.

---

## 3. Feature Extraction: ResNet-18 Backbone

### What ResNet-18 Produces
ResNet-18 was trained on ImageNet (1.2M images, 1000 classes). Its
layers progressively extract:

| Layer  | What it detects           | Feature map shape  |
|--------|---------------------------|--------------------|
| conv1  | Edges, colour gradients   | [64, 112, 112]     |
| layer1 | Textures, simple patterns | [64, 56, 56]       |
| layer2 | Parts (eyes, wheels)      | [128, 28, 28]      |
| layer3 | Object parts composed    | [256, 14, 14]      |
| layer4 | Full objects, scenes      | [512, 7, 7]        |
| avgpool| Global summary            | [512, 1, 1]        |

After `avgpool`, we flatten to get a single 512-dim vector that encodes
the image's high-level content.

### Why Freeze?
Freezing the backbone means:
- No gradients → no backpropagation → no weight updates
- The features are deterministic: same image → same vector, always
- No risk of overfitting (the backbone never adapts to our small dataset)

This is **transfer learning without fine-tuning**: we reuse ImageNet's
learned features as-is.

---

## 4. Distance Metrics

### 4.1 Cosine Similarity (Default)

```
sim(a, b) = (a · b) / (||a|| × ||b||)
```

**Intuition:** Measures the angle between two vectors. Two images with
the same "kind" of features (regardless of intensity) will have high
cosine similarity.

For L2-normalised vectors (||a|| = ||b|| = 1), this simplifies to:
```
sim(a, b) = a · b     (just a dot product!)
```

We convert to a **distance** for the k-NN search:
```
d_cos(a, b) = 1 - sim(a, b)
```

Range: [0, 2]. Identical vectors → 0. Orthogonal vectors → 1.

### 4.2 Euclidean Distance

```
d_euc(a, b) = ||a - b||_2 = √(Σ(a_i - b_i)²)
```

**Intuition:** The straight-line distance in feature space. Two images
with very similar feature vectors (same values, same magnitudes) will
have small Euclidean distance.

**Efficient computation** (avoids the O(B × N × D) explicit subtraction):
```
||a - b||² = ||a||² + ||b||² - 2(a · b)
```

This uses matrix multiplication (`a @ b.T`) which is highly optimised
on GPUs.

### When to Use Which?

| Metric    | Best when...                                           |
|-----------|--------------------------------------------------------|
| Cosine    | Feature magnitudes are less important than directions   |
| Euclidean | Absolute feature values matter (e.g., brightness level) |

For ResNet features, **cosine** typically works better because the
ReLU activations produce non-negative features with varying magnitudes,
and we care more about which features are active (direction) than how
strongly (magnitude).

---

## 5. Voting Mechanisms

### 5.1 Hard Voting (Simple Majority)
```
ŷ = argmax_c  |{i ∈ N_k(x) : y_i = c}|
```

Each of the k neighbours gets exactly 1 vote. The class with the most
votes wins. **Problem:** Ties are possible when k is even or when
neighbours are unevenly distributed.

### 5.2 Distance-Weighted Voting (What We Use)
```
ŷ = argmax_c  Σ_{i ∈ N_k(x)} w_i · 𝟙(y_i = c)

where  w_i = 1 / (d(x, x_i) + ε)
```

Closer neighbours get stronger votes. A training image at distance 0.01
has 100× more influence than one at distance 1.0.

**Why the ε (epsilon)?** If a query exactly matches a training image
(distance = 0), the weight would be 1/0 = ∞. Adding ε = 10⁻⁸ prevents
division by zero.

### Converting to "Probabilities"
We normalise the weighted votes to sum to 1:
```
P(class = c | x) = (Σ w_i · 𝟙(y_i = c)) / (Σ w_i)
```

This gives a soft output compatible with the DetectionResult format
(e.g., {"Real": 0.72, "AI-Generated": 0.28}).

---

## 6. The k Parameter

### Effect on the Decision Boundary

| k value | Behaviour                        | Risk              |
|---------|----------------------------------|--------------------|
| k = 1   | Nearest single neighbour         | Noisy, overfits    |
| k = 5   | Balanced (our default)           | Good generalisation|
| k = 20  | Very smooth decisions            | Under-sensitive    |
| k = N   | Global majority (ignores query)  | Useless            |

### How to Choose k
- **Odd k** avoids ties in binary classification
- **k ≈ √N** is a common heuristic (for N=400 samples, k ≈ 20)
- **Cross-validation** on the validation set is the proper method
- **k = 5** is a safe default that works well for most problems

---

## 7. Leave-One-Out Evaluation

### The Problem
If we evaluate k-NN on its own training set, each query trivially
matches itself (distance = 0), giving 100% accuracy. This is
meaningless.

### The Solution
**Leave-one-out (LOO):** For each training sample i, find its k
nearest neighbours EXCLUDING itself:

```
For each i in {1, ..., N}:
    N_k(x_i) = k nearest from {x_1, ..., x_{i-1}, x_{i+1}, ..., x_N}
    ŷ_i = majority_vote(labels of N_k(x_i))

LOO accuracy = |{i : ŷ_i = y_i}| / N
```

### Implementation Trick
Instead of N separate searches, we compute the full N×N distance matrix,
set the diagonal to ∞ (to exclude self-matches), and find the k smallest
in each row.

---

## 8. k-NN as XAI (Inherent Interpretability)

### Why k-NN is Self-Explaining
The CNN needs Grad-CAM to explain decisions. The ViT needs attention
rollout. The k-NN needs... nothing. Its explanation IS the algorithm:

> "I classified this image as AI-Generated because the 5 most similar
>  images in my training set were: [img_42, img_187, img_301, img_455,
>  img_512]. Of these, 4 were AI-Generated and 1 was Real."

The user can literally look at those 5 images and judge for themselves
whether the neighbours look similar and whether the classification
makes sense.

### Comparison of XAI Methods

| Detector | XAI Method           | Explains...                      |
|----------|----------------------|----------------------------------|
| CNN      | Grad-CAM             | Which PIXELS matter               |
| ViT      | Attention Rollout    | Which PATCHES matter              |
| k-NN     | Nearest Neighbours   | Which TRAINING IMAGES are similar |

These three perspectives are complementary:
- Grad-CAM: "The model focused on the jawline area"
- Attention rollout: "Patches 42 and 128 received the most attention"
- k-NN: "The 5 most similar training images were all AI-generated faces
  with similar lighting"

---

## 9. Complexity Analysis

| Operation        | Time Complexity           | Space Complexity |
|------------------|---------------------------|------------------|
| Feature extract  | O(D_backbone) per image   | O(512) per image |
| Store features   | O(N)                      | O(N × 512)       |
| Distance compute | O(N × 512) per query      | O(N)             |
| k-NN search      | O(N log k) per query      | O(k)             |
| **Total predict**| **O(N × 512)** per query  | O(N × 512)       |

**Key insight:** Inference time scales LINEARLY with the training set
size N. For N = 400 (our typical dataset), this is instant. For N = 1M,
you'd need approximate nearest-neighbour methods (FAISS, Annoy) — but
we're not there.

### Comparison to CNN and ViT

| Model  | Training cost      | Inference cost    | Memory         |
|--------|--------------------|-------------------|----------------|
| CNN    | O(epochs × N)      | O(1) forward pass | ~470K params   |
| ViT    | O(epochs × N)      | O(1) forward pass | ~5.5M params   |
| k-NN   | O(N) feature ext   | O(N) search       | N × 512 floats |
