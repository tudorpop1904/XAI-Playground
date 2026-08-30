"""
core/detectors/kmc.py
======================
k-Means Clustering (KMC) detector for deepfake image classification.

WHAT IS KMC?
------------
k-Means Clustering is an unsupervised/representation-transfer learning
algorithm that partitions latent feature representations into k clusters
(centroids).

Unlike purely supervised detectors, KMC groups images based on their
geometric proximity in the transferred latent space. During training,
it clusters feature vectors extracted by a pre-trained backbone (ResNet-18,
FTL-CNN, or ViT) and assigns the majority label to each cluster.

During inference, KMC assigns each input image to the nearest cluster
centroid and returns confidence based on the distance ratio.

CROSS-ARCHITECTURE REPRESENTATION TRANSFER:
-------------------------------------------
    Image [3, H, W]
         │
         ▼
    ┌───────────────────────────────────┐
    │  Frozen Backbone                  │
    │  (ResNet-18 / FTL-CNN / ViT)      │
    └─────────────────┬─────────────────┘
                      │
                      ▼
    Latent Vector z ∈ R^D (e.g. 512 / 256 / 192)
                      │
                      ▼
    ┌───────────────────────────────────┐
    │  k-Means Centroids (Learned)      │
    │  C_1 (Real), C_2 (AI-Generated)   │
    └─────────────────┬─────────────────┘
                      │
                      ▼
    Distance-Weighted Cluster Assignment → DetectionResult
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import AbstractBaseDetector, logger
from core.detectors.feature_extractors import get_feature_extractor
from core.results.detection import DetectionResult
from core.utils.metrics import track_hardware

# -- Device-agnostic setup ------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {0: "Real", 1: "AI-Generated"}


class KMCDetector(AbstractBaseDetector):
    """
    k-Means Clustering detector for deepfake classification.

    Parameters
    ----------
    k : int
        Number of clusters to learn (default: 2 for Real vs AI).
    metric : str
        Distance metric: "cosine" or "euclidean" (default: "cosine").
    backbone : str
        Feature extraction backbone: "resnet18", "ftl_cnn", or "vit".
    max_iters : int
        Maximum iterations for centroid convergence (default: 100).
    """

    index = 1

    def __init__(
        self,
        k: int = 2,
        metric: Literal["cosine", "euclidean"] = "cosine",
        num_classes: int = 2,
        backbone: str = "resnet18",
        max_iters: int = 100,
    ) -> None:
        super().__init__(name=f"KMC_{KMCDetector.index}")
        KMCDetector.index += 1

        self.k = k
        self.metric = metric
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.max_iters = max_iters

        # Feature extractor
        self.backbone = get_feature_extractor(
            backbone, normalize=(metric == "cosine")
        )

        # Buffers for centroids and cluster labels
        self.register_buffer("centroids", torch.empty(0, self.backbone.feat_dim))
        self.register_buffer("cluster_labels", torch.empty(0, dtype=torch.long))

        self.to(device)

    def get_target_layer(self) -> Optional[nn.Module]:
        return getattr(self.backbone, "target_layer", None)

    # ── Feature Extraction & Training ────────────────────────────────

    @torch.inference_mode()
    def fit(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Extracts features from the dataloader and runs k-Means clustering.
        """
        self.eval()
        self.backbone.to(device)

        all_features = []
        all_labels = []

        logger.info(f"[{self.name}] Extracting features using backbone '{self.backbone_name}'...")
        for images, labels in dataloader:
            images = images.to(device)
            feats = self.backbone(images)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())

        X = torch.cat(all_features, dim=0).to(device)  # [N, D]
        y = torch.cat(all_labels, dim=0).to(device)    # [N]
        N, D = X.shape

        # Initialize centroids randomly from data points
        rand_indices = torch.randperm(N)[:self.k]
        centroids = X[rand_indices].clone()

        # k-Means iteration
        for it in range(self.max_iters):
            if self.metric == "cosine":
                # Cosine similarity: max dot product (since vectors are L2-normalized)
                sims = torch.mm(X, centroids.t())  # [N, k]
                cluster_assignments = sims.argmax(dim=1)
            else:
                # Euclidean distance
                dists = torch.cdist(X, centroids)  # [N, k]
                cluster_assignments = dists.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for c in range(self.k):
                mask = (cluster_assignments == c)
                if mask.sum() > 0:
                    new_centroids[c] = X[mask].mean(dim=0)
                    if self.metric == "cosine":
                        new_centroids[c] = F.normalize(new_centroids[c], p=2, dim=0)
                else:
                    new_centroids[c] = X[torch.randint(0, N, (1,))[0]]

            shift = (new_centroids - centroids).norm().item()
            centroids = new_centroids
            if shift < 1e-4:
                logger.info(f"[{self.name}] k-Means converged at iteration {it+1}.")
                break

        # Map each cluster to majority ground-truth label
        cluster_labels = torch.zeros(self.k, dtype=torch.long, device=device)
        for c in range(self.k):
            mask = (cluster_assignments == c)
            if mask.sum() > 0:
                cluster_labels[c] = torch.mode(y[mask]).values.item()
            else:
                cluster_labels[c] = c % self.num_classes

        self.centroids = centroids
        self.cluster_labels = cluster_labels

        logger.info(f"[{self.name}] Training complete. Cluster mappings: {cluster_labels.tolist()}")
        return {"centroids_learned": self.k, "feature_dim": D}

    # ── Inference Pass ────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning pseudo-logits based on distance to centroids.
        """
        if self.centroids.numel() == 0:
            raise RuntimeError(f"[{self.name}] Model not fitted. Call fit() first.")

        feats = self.backbone(x)  # [B, D]
        if self.metric == "cosine":
            sims = torch.mm(feats, self.centroids.t())  # [B, k]
            # Convert similarities into class logits
            logits = torch.zeros(x.shape[0], self.num_classes, device=x.device)
            for c in range(self.k):
                cls = self.cluster_labels[c].item()
                logits[:, cls] = torch.maximum(logits[:, cls], sims[:, c] * 5.0)
            return logits
        else:
            dists = torch.cdist(feats, self.centroids)  # [B, k]
            # Invert distance to get positive logits
            scores = 1.0 / (dists + 1e-5)
            logits = torch.zeros(x.shape[0], self.num_classes, device=x.device)
            for c in range(self.k):
                cls = self.cluster_labels[c].item()
                logits[:, cls] = torch.maximum(logits[:, cls], scores[:, c])
            return logits

    @track_hardware
    def predict(self, image_path: str) -> DetectionResult:
        """
        Classifies an image and returns a DetectionResult.
        """
        self.eval()
        if isinstance(image_path, str):
            image_tensor = torch.load(image_path, weights_only=False)
        else:
            image_tensor = image_path

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(device)
        logits = self.forward(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

        probabilities = {
            CLASS_NAMES[i]: round(probs[i].item(), 4)
            for i in range(self.num_classes)
        }

        return DetectionResult(
            ai_deepfake=(predicted_idx == 1),
            confidence=round(confidence, 4),
            image=str(image_path),
            model=self,
            returned_obj=probabilities,
        )

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        checkpoint_dir: Optional[str | Path] = None,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
        device_override: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """
        Trains the KMC model via fit() to adhere to AbstractBaseDetector interface.
        """
        self.fit(train_loader)
        return {"epochs": [1], "train_loss": [0.0], "val_loss": [0.0]}