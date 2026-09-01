"""
core/utils/custom_dataset.py
=============================
Custom PyTorch Dataset for indexed file-path loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from core.utils.logger import get_logger

logger = get_logger(__name__)


class FileListDataset(Dataset):
    """
    A custom PyTorch Dataset that loads images directly from absolute file paths.
    Avoids duplicating files into a standard ImageFolder directory structure.
    """

    def __init__(
        self,
        index_data: dict[str, list[str]],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """
        Parameters
        ----------
        index_data : dict[str, list[str]]
            Dictionary with "real" and "fake" keys containing lists of absolute file paths.
        transform : Optional[Callable[[Image.Image], torch.Tensor]]
            PyTorch torchvision transformations to apply to each image.
        """
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        # Class mapping: 0 = Real, 1 = AI-Generated (Fake)
        for path in index_data.get("real", []):
            self.samples.append((str(path), 0))

        for path in index_data.get("fake", []):
            self.samples.append((str(path), 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        path, label = self.samples[idx]

        try:
            # Convert to RGB to handle grayscale (1-ch) or RGBA (4-ch) images uniformly
            image = Image.open(path).convert("RGB")
        except Exception as err:
            logger.warning(f"Error loading image '{path}': {err}. Falling back to blank tensor.")
            image = Image.new("RGB", (128, 128), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
