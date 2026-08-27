from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class FileListDataset(Dataset):
    """
    A custom PyTorch Dataset that loads images directly from absolute file paths.
    This avoids the need to duplicate files into a standard ImageFolder directory structure.
    """
    def __init__(self, index_data: dict, transform=None):
        """
        index_data: dict with "real" and "fake" keys, each containing a list of absolute file paths.
        """
        self.transform = transform
        self.samples = []
        
        # We assign class 0 to real, class 1 to fake (AI)
        # This matches our CNN/ViT output mapping
        for path in index_data.get("real", []):
            self.samples.append((path, 0))
            
        for path in index_data.get("fake", []):
            self.samples.append((path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            # Convert to RGB in case of grayscale or RGBA images
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Fallback to a blank image if loading fails
            image = Image.new('RGB', (128, 128))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
