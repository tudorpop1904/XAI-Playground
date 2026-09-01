"""
core/detectors/knn.py
======================
k-Nearest Neighbours (k-NN) detector for deepfake image classification.

WHAT IS k-NN?
--------------
k-NN is the simplest classifier that exists. It has NO learnable
parameters. It memorises the entire training set and classifies new
images by finding the k most similar training images and taking a
majority vote.

This makes it fundamentally different from our CNN and ViT:

  ┌──────────────────┬──────────────┬──────────────┬──────────────┐
  │     Property     │     CNN      │     ViT      │     k-NN     │
  ├──────────────────┼──────────────┼──────────────┼──────────────┤
  │ Type             │ Parametric   │ Parametric   │Non-parametric│
  │ Learns weights?  │ Yes (~470K)  │ Yes (~5.5M)  │ No (0)       │
  │ Training         │ Backprop     │ Backprop     │ Store vectors│
  │ Inference speed  │ O(1) forward │ O(1) forward │ O(N) search  │
  │ XAI method       │ Grad-CAM     │ Attn rollout │ Show k nbrs  │
  │ Decision boundary│ Smooth       │ Smooth       │ Jagged       │
  └──────────────────┴──────────────┴──────────────┴──────────────┘

WHY INCLUDE k-NN?
-----------------
1. **Research baseline** — k-NN with a frozen ImageNet backbone tells
   you how much of deepfake detection is "just" feature similarity
   vs. learned decision boundaries. If k-NN does nearly as well as
   the CNN, the CNN isn't learning anything novel.

2. **Inherently interpretable XAI** — The "explanation" is literally
   the k most similar training images. No heatmaps, no gradients,
   no attention rollout needed. A human can look at the neighbors
   and immediately understand WHY the model made its decision.

3. **Non-parametric comparison** — Your thesis can now compare three
   fundamentally different paradigms: local features (CNN), global
   attention (ViT), and instance-based similarity (k-NN).

THE TRICK: FEATURE EXTRACTION
-------------------------------
Raw pixel comparison doesn't work (curse of dimensionality — all
images are roughly equidistant in 224x224x3 = ~150K-dimensional
pixel space). Instead, we use a FROZEN pretrained ResNet-18 as a
feature extractor:

    Image [3, 224, 224]
         │
         ▼
    ┌───────────────────────────────┐
    │  Frozen ResNet-18 Backbone    │
    │  (pretrained on ImageNet)     │
    │  Remove the final FC layer    │
    │  Keep everything else         │
    └──────────────┬────────────────┘
                   │
                   ▼
    Feature vector [512]
         │
         ▼
    ┌───────────────────────────────┐
    │  k-NN Lookup                  │
    │  Compare to all stored        │
    │  training feature vectors     │
    │  using cosine similarity      │
    └──────────────┬────────────────┘
                   │
                   ▼
    ┌───────────────────────────────┐
    │  Majority Vote                │
    │  Among the k nearest,         │
    │  which class has more votes?  │
    └──────────────┬────────────────┘
                   │
                   ▼
    Prediction: "Real" or "AI-Generated"

PAPER REFERENCE
----------------
- T. Cover & P. Hart (1967).
  "Nearest Neighbor Pattern Classification"
  IEEE Transactions on Information Theory, 13(1), 21-27.
- For ResNet backbone: K. He et al. (CVPR 2016).
  "Deep Residual Learning for Image Recognition"
"""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn.functional as F

from core.utils.metrics import track_hardware
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from .base import AbstractBaseDetector, logger

from core.results.detection import DetectionResult


# -- Device-agnostic setup ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -- Class label mapping ---------------------------------------------------
CLASS_NAMES = {0: "Real", 1: "AI-Generated"}


# =========================================================================
#  FEATURE EXTRACTOR (Frozen ResNet-18 Backbone)
# =========================================================================


class ResNetFeatureExtractor(nn.Module):
    """
    A frozen ResNet-18 used purely as a feature extractor.

    WHAT THIS DOES
    ---------------
    ResNet-18 was trained on ImageNet (1.2M images, 1000 classes).
    Its convolutional layers have learned to detect edges, textures,
    shapes, and high-level object features. By removing the final
    classification head and keeping everything else, we get a
    universal feature extractor that maps any image to a 512-dim
    vector.

    WHY FREEZE?
    We don't want to fine-tune the backbone. The k-NN classifier is
    non-parametric — its power comes from the QUALITY of the feature
    space, not from learned decision boundaries. The ImageNet features
    are already rich enough for our purpose.

    Freezing also means:
      - No gradient computation (faster inference)
      - No risk of overfitting to our small dataset
      - Deterministic feature extraction (same image always gives
        the same vector)

    ARCHITECTURE
    -------------
    ResNet-18 (full):      ResNet-18 (as feature extractor):
    conv1                  conv1
    bn1 + relu + maxpool   bn1 + relu + maxpool
    layer1 (2 blocks)      layer1 (2 blocks)
    layer2 (2 blocks)      layer2 (2 blocks)
    layer3 (2 blocks)      layer3 (2 blocks)
    layer4 (2 blocks)      layer4 (2 blocks)
    avgpool                avgpool
    fc (512 -> 1000)       [REMOVED]
                           Output: [B, 512]

    Parameters
    ----------
    normalize : bool
        If True, L2-normalise the output features. This is important
        when using cosine similarity (L2-normed vectors turn dot
        product into cosine similarity).
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

        # Load pretrained ResNet-18
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final FC layer — keep everything else
        # nn.Sequential unpacks all children EXCEPT the last (fc)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        # Freeze ALL parameters — no gradients, no updates
        for param in self.features.parameters():
            param.requires_grad = False

        # Store the last conv layer reference for Grad-CAM fallback
        self.target_layer = backbone.layer4

        # Feature dimension
        self.feat_dim = 512

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract a 512-dimensional feature vector from each image.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Shape [B, 512], optionally L2-normalised.
        """
        # Resize to 224x224 if needed (ResNet's expected input)
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        # Forward through backbone -> [B, 512, 1, 1]
        features = self.features(x)

        # Flatten: [B, 512, 1, 1] -> [B, 512]
        features = features.flatten(1)

        # Optional L2 normalisation
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)

        return features


# =========================================================================
#  THE k-NN DETECTOR
# =========================================================================


class KNNDetector(AbstractBaseDetector):
    """
    k-Nearest Neighbours detector for deepfake classification.

    Uses a frozen ResNet-18 backbone to extract 512-dim feature vectors,
    then classifies by finding the k closest training vectors.

    This is fundamentally different from the CNN and ViT:
      - NO learnable parameters (the backbone is frozen)
      - "Training" = extracting and storing feature vectors
      - "Inference" = comparing the query to all stored vectors
      - "Explanation" = showing the k most similar training images

    DISTANCE METRICS
    -----------------
    Two options:

    1. COSINE SIMILARITY (default):
         sim(a, b) = (a . b) / (||a|| * ||b||)

       Range: [-1, 1] (with L2-normed vectors, always in [0, 1]).
       Measures the ANGLE between vectors, ignoring magnitude.
       Good for high-dimensional spaces because it's less sensitive
       to the curse of dimensionality.

    2. EUCLIDEAN DISTANCE:
         dist(a, b) = ||a - b||_2 = sqrt(sum((a_i - b_i)^2))

       Range: [0, infinity).
       Measures the MAGNITUDE of the difference vector.
       More sensitive to outliers but can capture absolute differences
       that cosine misses.

    SOFT VOTING
    ------------
    Instead of hard majority voting (each neighbor gets 1 vote), we
    use distance-weighted voting. Closer neighbors get stronger votes:

        weight_i = 1 / (distance_i + epsilon)

    This prevents ties and gives more influence to very similar images.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to consider (default: 5).
    metric : str
        Distance metric: "cosine" or "euclidean" (default: "cosine").

    Example
    -------
    >>> detector = KNNDetector(k=5, metric="cosine")
    >>> detector.fit(train_loader)        # Extract + store features
    >>> result = detector.predict(image)  # Find 5 nearest, vote
    """

    # -- Class-level counter for auto-naming instances --------------------
    index = 1

    def __init__(
        self,
        k: int = 5,
        metric: Literal["cosine", "euclidean"] = "cosine",
        num_classes: int = 2,
        backbone: str = "resnet18",
    ) -> None:
        super().__init__(name=f"KNN_{KNNDetector.index}")
        KNNDetector.index += 1

        # Configuration
        self.k = k
        self.metric = metric
        self.num_classes = num_classes
        self.backbone_name = backbone

        # ----------------------------------------------------------------
        # STAGE 0: Feature Extractor (Representation Transfer Backbone)
        # ----------------------------------------------------------------
        # Supports: 'resnet18' (ImageNet), 'ftl_cnn' (Forensic), 'vit' (Attention)
        # ----------------------------------------------------------------
        from core.detectors.feature_extractors import get_feature_extractor
        self.backbone = get_feature_extractor(
            backbone, normalize=(metric == "cosine")
        )

        # ----------------------------------------------------------------
        # STAGE 1: Feature Store (populated during fit())
        # ----------------------------------------------------------------
        #
        # These tensors hold the extracted features and labels from the
        # training set. They're registered as buffers (not parameters)
        # so that:
        #   1. They're saved/loaded with the model state dict
        #   2. They're moved to the correct device with .to(device)
        #   3. They're NOT included in optimizer.parameters()
        #
        # We initialise them as empty and populate them in fit().
        # ----------------------------------------------------------------
        self.register_buffer(
            "train_features", torch.empty(0, self.backbone.feat_dim)
        )
        self.register_buffer("train_labels", torch.empty(0, dtype=torch.long))

        # ----------------------------------------------------------------
        # STAGE 2: Training Image References (for XAI)
        # ----------------------------------------------------------------
        #
        # We store the ACTUAL training images so that when the user asks
        # "why did you classify this as AI?", we can show them the k
        # most similar training images. This is the k-NN equivalent of
        # a heatmap — instead of highlighting pixels, we show examples.
        #
        # NOTE: This is stored as a plain list, not a buffer, because
        # tensors of varying sizes can't be stacked into a single tensor.
        # For large datasets, consider storing paths instead.
        # ----------------------------------------------------------------
        self.train_images: list[torch.Tensor] = []

        # Flag to check if the model has been fitted
        self._is_fitted = False

    # =====================================================================
    #  Forward Pass
    # =====================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features and compute class logits via k-NN voting.

        This method fulfils the AbstractBaseDetector contract. For a
        standard nn.Module, forward() runs the network. For k-NN,
        forward() extracts features and performs the nearest-neighbour
        lookup.

        If the model hasn't been fitted yet (no stored training data),
        this returns zero logits (the model hasn't "learned" anything).

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Soft logits of shape [B, num_classes]. These are NOT
            true logits (they're vote proportions in [0, 1]), but
            they're compatible with CrossEntropyLoss and softmax.
        """
        B = x.shape[0]

        # Edge case: not fitted yet — return zero logits
        if not self._is_fitted or self.train_features.shape[0] == 0:
            logger.warning(f"[KNN] Not fitted. Returning zero logits.")
            return torch.zeros(B, self.num_classes, device=x.device)

        # Step 1: Extract features from the input batch
        logger.info(f"[KNN] forward pass started. Input shape: {x.shape}")
        query_features = self.backbone(x)  # [B, 512]
        logger.info(f"[KNN] Backbone extracted features shape: {query_features.shape}")

        # Step 2: Compute distances to all training features
        distances = self._compute_distances(
            query_features, self.train_features
        )  # [B, N_train]

        # Step 3: Find k nearest neighbours
        # torch.topk with largest=False gives the k SMALLEST distances
        # (i.e., the k MOST similar training samples)
        k = min(self.k, self.train_features.shape[0])
        top_distances, top_indices = torch.topk(
            distances, k=k, dim=1, largest=False
        )  # Both: [B, k]

        # Step 4: Get the labels of the k nearest neighbours
        neighbour_labels = self.train_labels[top_indices]  # [B, k]

        # Step 5: Distance-weighted voting
        #
        # Each neighbour votes for its class, but closer neighbours
        # get stronger votes:
        #   weight_i = 1 / (distance_i + epsilon)
        #
        # This is better than hard voting because:
        #   - Prevents ties
        #   - A very close neighbour is more reliable than a far one
        #   - Produces smoother probability estimates
        weights = 1.0 / (top_distances + 1e-8)  # [B, k]

        # Aggregate votes per class
        logits = torch.zeros(B, self.num_classes, device=x.device)
        for c in range(self.num_classes):
            # Mask: 1 where the neighbour is class c, 0 otherwise
            class_mask = (neighbour_labels == c).float()  # [B, k]
            # Sum of weights for class c
            logits[:, c] = (weights * class_mask).sum(dim=1)

        # Normalise to sum to 1 (makes them behave like probabilities)
        logits = logits / logits.sum(dim=1, keepdim=True).clamp(min=1e-8)
        logger.info(f"[KNN] forward pass ended. Logits shape: {logits.shape}")

        return logits

    # =====================================================================
    #  Distance Computation
    # =====================================================================

    def _compute_distances(
        self,
        query: torch.Tensor,
        gallery: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distances between query and gallery vectors.

        Parameters
        ----------
        query : torch.Tensor
            Shape [B, D] — the images to classify.
        gallery : torch.Tensor
            Shape [N, D] — the stored training features.

        Returns
        -------
        torch.Tensor
            Shape [B, N] — distance from each query to each gallery item.
            LOWER = more similar (regardless of metric).
        """
        if self.metric == "cosine":
            # Cosine similarity: dot product of L2-normed vectors
            # Range: [0, 1] (since vectors are L2-normed and non-negative
            #         in practice after ReLU features)
            #
            # We return (1 - similarity) as distance so that
            # LOWER = more similar.
            similarity = query @ gallery.T  # [B, N]
            return 1.0 - similarity

        elif self.metric == "euclidean":
            # Euclidean distance: ||a - b||_2
            #
            # Efficient computation using the expansion:
            #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
            #
            # This avoids the O(B * N * D) explicit subtraction and
            # instead uses matrix multiplication O(B * N * D) with
            # better memory access patterns.
            query_sq = (query ** 2).sum(dim=1, keepdim=True)   # [B, 1]
            gallery_sq = (gallery ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
            cross = query @ gallery.T                            # [B, N]
            dist_sq = query_sq + gallery_sq - 2 * cross
            # Clamp to avoid negative values from floating-point errors
            return torch.sqrt(dist_sq.clamp(min=0))

        else:
            raise ValueError(
                f"Unknown metric '{self.metric}'. Use 'cosine' or 'euclidean'."
            )

    # =====================================================================
    #  Training (Feature Extraction + Storage)
    # =====================================================================

    @torch.inference_mode()
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 1,       # Ignored — k-NN doesn't iterate
        lr: float = 0.0,       # Ignored — no optimiser
        **kwargs,
    ) -> dict[str, list[float]]:
        """
        "Train" the k-NN detector by extracting and storing features
        from the training dataset.

        This is NOT training in the neural network sense. There are no
        gradients, no optimiser, no loss function. We simply:
          1. Pass every training image through the frozen backbone
          2. Store the resulting 512-dim feature vector
          3. Store the corresponding label

        This is a single pass through the data — the `epochs` parameter
        is accepted for API compatibility but ignored.

        Parameters
        ----------
        train_loader : DataLoader
            Training data yielding (images, labels) batches.
        val_loader : DataLoader | None
            If provided, we evaluate accuracy on the validation set
            after fitting.
        epochs : int
            Ignored (k-NN is a single-pass algorithm).
        lr : float
            Ignored (no optimiser).

        Returns
        -------
        dict[str, list[float]]
            Training "history" with keys matching the CNN/ViT format.
            train_loss is always 0 (there is no loss function).
            train_acc is the leave-one-out accuracy on the training set.
        """
        self.to(device)
        self.eval()  # Backbone is always in eval mode

        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_images: list[torch.Tensor] = []

        print(f"[{self.name}] Extracting features from training set...")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)

            # Extract features: [B, 3, H, W] -> [B, 512]
            features = self.backbone(images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            # Store images for XAI (on CPU to save GPU memory)
            for img in images:
                all_images.append(img.cpu())

            if (batch_idx + 1) % 10 == 0:
                n = sum(f.shape[0] for f in all_features)
                print(f"  Processed {n} images...")

        # Concatenate into single tensors
        self.train_features = torch.cat(all_features, dim=0).to(device)
        self.train_labels = torch.cat(all_labels, dim=0).to(device)
        self.train_images = all_images
        self._is_fitted = True

        n_total = self.train_features.shape[0]
        n_real = (self.train_labels == 0).sum().item()
        n_ai = (self.train_labels == 1).sum().item()

        print(
            f"[{self.name}] Stored {n_total} feature vectors "
            f"({n_real} Real, {n_ai} AI-Generated)"
        )

        # Build history dict (compatible with CNN/ViT format)
        history: dict[str, list[float]] = {
            "train_loss": [0.0],  # No loss function
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # Compute training accuracy (how well k-NN classifies its own
        # training data — this uses leave-one-out to avoid trivially
        # matching each image to itself)
        train_acc = self._leave_one_out_accuracy()
        history["train_acc"].append(train_acc)
        print(f"[{self.name}] Leave-one-out train accuracy: {train_acc:.4f}")

        # Evaluate on validation set if provided
        if val_loader is not None:
            val_loss, val_acc = self._evaluate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"[{self.name}] Validation accuracy: {val_acc:.4f}")

        return history

    def _leave_one_out_accuracy(self, max_samples: int = 2000) -> float:
        """
        Fast chunked leave-one-out accuracy without O(N^2) memory explosion.
        """
        N = self.train_features.shape[0]
        if N < self.k + 1:
            return 0.0

        # Sample up to max_samples for instantaneous validation
        if N > max_samples:
            indices = torch.randperm(N)[:max_samples]
            queries = self.train_features[indices]
            query_labels = self.train_labels[indices]
        else:
            indices = torch.arange(N)
            queries = self.train_features
            query_labels = self.train_labels

        correct = 0
        total = queries.shape[0]
        chunk_size = 500

        for i in range(0, total, chunk_size):
            q_chunk = queries[i : i + chunk_size]
            q_idx_chunk = indices[i : i + chunk_size]
            
            # Compute distance chunk [C, N]
            dists = self._compute_distances(q_chunk, self.train_features)
            
            # Mask out self-distance
            for r in range(q_chunk.shape[0]):
                dists[r, q_idx_chunk[r]] = float("inf")
                
            # Top-k nearest neighbours
            k = min(self.k, N - 1)
            _, top_k_idx = torch.topk(dists, k=k, dim=1, largest=False)
            nbr_labels = self.train_labels[top_k_idx]  # [C, k]
            
            # Majority vote
            mode_labels, _ = torch.mode(nbr_labels, dim=1)
            correct += (mode_labels == query_labels[i : i + chunk_size]).sum().item()

        return correct / total if total > 0 else 0.0

    # =====================================================================
    #  Validation / Evaluation
    # =====================================================================

    @torch.inference_mode()
    def _evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module | None = None,
    ) -> tuple[float, float]:
        """
        Evaluate the k-NN on a dataset.

        Parameters
        ----------
        data_loader : DataLoader
            Yields (images, labels) batches.
        criterion : nn.Module | None
            Ignored (k-NN has no loss function). Accepted for API
            compatibility with the CNN/ViT _evaluate() signature.

        Returns
        -------
        tuple[float, float]
            (average_loss, accuracy). Loss is always 0.0.
        """
        self.eval()
        correct = 0
        total = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = self(images)  # [B, num_classes]
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return 0.0, accuracy

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

        # Forward pass -> soft logits [1, num_classes]
        logits = self(x)

        # These are already normalised to [0, 1] summing to 1
        probs = logits.squeeze(0)  # [num_classes]

        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

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
    #  XAI: Nearest Neighbour Visualisation
    # =====================================================================

    @torch.inference_mode()
    def get_nearest_neighbors(
        self,
        image_tensor: torch.Tensor,
        n: int | None = None,
    ) -> list[dict]:
        """
        Find the k nearest training images to the query image.

        This IS the k-NN's explainability method. Instead of a heatmap,
        we show the user: "Here are the k training images most similar
        to your query, and their labels." A human can then assess
        whether the classification makes sense.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Single image tensor of shape [C, H, W], values in [0, 1].
        n : int | None
            Number of neighbours to return. Defaults to self.k.

        Returns
        -------
        list[dict]
            Each dict contains:
              - "image": torch.Tensor [C, H, W]
              - "label": str ("Real" or "AI-Generated")
              - "distance": float
              - "rank": int (1 = most similar)
        """
        if not self._is_fitted:
            return []

        n = n or self.k
        n = min(n, self.train_features.shape[0])

        self.to(device)

        # Extract query features
        if image_tensor.dim() == 3:
            x = image_tensor.unsqueeze(0).to(device)
        else:
            x = image_tensor.to(device)
        query_feat = self.backbone(x)  # [1, 512]

        # Compute distances to all training features
        distances = self._compute_distances(
            query_feat, self.train_features
        ).squeeze(0)  # [N_train]

        # Find n nearest
        top_distances, top_indices = torch.topk(
            distances, k=n, largest=False
        )

        # Build result list
        neighbors = []
        for rank, (idx, dist) in enumerate(
            zip(top_indices.tolist(), top_distances.tolist()), start=1
        ):
            label_idx = self.train_labels[idx].item()
            neighbors.append({
                "image": self.train_images[idx],
                "label": CLASS_NAMES[label_idx],
                "distance": round(dist, 6),
                "rank": rank,
            })

        return neighbors

    # =====================================================================
    #  XAI Hook: Target Layer for Grad-CAM
    # =====================================================================

    def get_target_layer(self) -> Optional[nn.Module]:
        """
        Return the last convolutional block of the ResNet backbone.

        While k-NN's native XAI is nearest-neighbour visualisation,
        we can also run Grad-CAM on the backbone's last conv layer
        to see what IMAGE REGIONS the feature extractor focuses on.
        This is a secondary XAI method, not the primary one.

        Returns
        -------
        nn.Module
            ResNet-18's layer4 (the last residual block).
        """
        return self.backbone.target_layer

    # =====================================================================
    #  Utility: Model Summary
    # =====================================================================

    def summary(self, input_size: tuple[int, int] = (224, 224)) -> str:
        """
        Print a human-readable summary of the k-NN detector.

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
        n_stored = self.train_features.shape[0] if self._is_fitted else 0

        backbone_params = sum(
            p.numel() for p in self.backbone.parameters()
        )

        lines = [
            f"{'='*65}",
            f"  {self.name} -- k-Nearest Neighbours Detector",
            f"  k={self.k}, metric={self.metric}",
            f"{'='*65}",
            f"  {'Component':<30} {'Details':<30}",
            f"  {'-'*55}",
            f"  {'Backbone':<30} ResNet-18 (frozen)",
            f"  {'Backbone params':<30} {backbone_params:,} (0 trainable)",
            f"  {'Feature dimension':<30} {self.backbone.feat_dim}",
            f"  {'Stored training vectors':<30} {n_stored:,}",
            f"  {'Distance metric':<30} {self.metric}",
            f"  {'Neighbours (k)':<30} {self.k}",
            f"  {'Input size':<30} [B, 3, {h}, {w}]",
            f"  {'Fitted':<30} {self._is_fitted}",
            f"  {'-'*55}",
            f"  Learnable parameters: 0",
            f"  Total parameters: {backbone_params:,} (all frozen)",
            f"{'='*65}",
        ]

        summary_str = "\n".join(lines)
        print(summary_str)
        return summary_str

    def get_target_layer(self) -> Optional[nn.Module]:
        """
        Return the target layer of the underlying feature extraction backbone.
        """
        return getattr(self.backbone, "target_layer", None)
