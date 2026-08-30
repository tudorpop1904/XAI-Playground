"""
core/detectors/feature_extractors.py
====================================
Modular Feature Extractors for Representation Transfer Learning.

Enables non-parametric and clustering detectors (k-NN, KMC) to extract
latent representation vectors from distinct pre-trained backbones:
  1. ResNet-18 (ImageNet Pretrained CNN — 512 dims)
  2. FTL-CNN (Custom Forensic CNN with FFT/LBP/Sobel — 256 dims)
  3. ViT (Vision Transformer CLS Token Embedding — 192/384/768 dims)

This implements Cross-Architecture Representation Transfer:
    Image [3, H, W]  ──▶  [ Frozen Backbone ]  ──▶  Latent Vector z ∈ R^D  ──▶  k-NN / KMC
"""

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from core.utils.image_features import compute_fft, compute_lbp, compute_sobel


# =========================================================================
#  1. ResNet-18 Feature Extractor (ImageNet Pretrained)
# =========================================================================

class ResNetFeatureExtractor(nn.Module):
    """
    Frozen ResNet-18 backbone pre-trained on ImageNet (512-dim output).
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.feat_dim = 512

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

        for param in self.features.parameters():
            param.requires_grad = False

        self.target_layer = backbone.layer4

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        features = self.features(x).flatten(1)
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        return features


# =========================================================================
#  2. FTL Forensic CNN Feature Extractor (FFT + LBP + Sobel)
# =========================================================================

class FTLFeatureExtractor(nn.Module):
    """
    Frozen FTL-CNN feature extractor (256-dim output).
    Extracts forensic frequency, texture, and edge features.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.feat_dim = 256

        # Adapter for 6-channel forensic input -> 3 channels
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # 4 Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        for param in self.parameters():
            param.requires_grad = False

        self.target_layer = self.conv4

    def _enrich_features(self, x: torch.Tensor) -> torch.Tensor:
        """Appends FFT, LBP, and Sobel channels to RGB input."""
        device = x.device
        b, _, h, w = x.shape
        extra_channels = []

        for i in range(b):
            img_np = x[i].permute(1, 2, 0).cpu().numpy()
            fft_ch = torch.from_numpy(compute_fft(img_np)).float().unsqueeze(0)
            lbp_ch = torch.from_numpy(compute_lbp(img_np)).float().unsqueeze(0)
            sobel_ch = torch.from_numpy(compute_sobel(img_np)).float().unsqueeze(0)
            combined = torch.cat([fft_ch, lbp_ch, sobel_ch], dim=0)
            extra_channels.append(combined.unsqueeze(0))

        extra_tensor = torch.cat(extra_channels, dim=0).to(device)
        return torch.cat([x, extra_tensor], dim=1)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x_rich = self._enrich_features(x)
        out = self.channel_adapter(x_rich)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        features = self.global_pool(out).flatten(1)

        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
        return features


# =========================================================================
#  3. Vision Transformer Feature Extractor (CLS Token)
# =========================================================================

class ViTFeatureExtractor(nn.Module):
    """
    Frozen Vision Transformer backbone extracting the global CLS token representation.
    """

    def __init__(self, embed_dim: int = 192, depth: int = 4, num_heads: int = 4, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.feat_dim = embed_dim

        from core.detectors.vit import VisionTransformer
        self.vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=2,
        )

        for param in self.vit.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Extract CLS token from Transformer blocks
        x_proj = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x_tokens = torch.cat((cls_tokens, x_proj), dim=1)
        x_tokens = self.vit.pos_drop(x_tokens + self.vit.pos_embed)

        for block in self.vit.blocks:
            x_tokens = block(x_tokens)

        x_tokens = self.vit.norm(x_tokens)
        cls_out = x_tokens[:, 0]  # [B, embed_dim]

        if self.normalize:
            cls_out = F.normalize(cls_out, p=2, dim=1)
        return cls_out


# =========================================================================
#  Factory Function
# =========================================================================

def get_feature_extractor(
    backbone_type: Literal["resnet18", "ftl_cnn", "vit"] = "resnet18",
    normalize: bool = True
) -> nn.Module:
    """
    Factory function for selecting feature extractors for Representation Transfer.
    """
    bb = backbone_type.lower()
    if bb in ["resnet", "resnet18", "resnet-18"]:
        return ResNetFeatureExtractor(normalize=normalize)
    elif bb in ["ftl", "ftl_cnn", "cnn"]:
        return FTLFeatureExtractor(normalize=normalize)
    elif bb in ["vit", "transformer"]:
        return ViTFeatureExtractor(normalize=normalize)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Choose 'resnet18', 'ftl_cnn', or 'vit'.")
