from abc import ABC, abstractmethod
import torch

class BaseEnhancer(ABC):
    """
    Base interface for all AI image enhancers.
    """
    @abstractmethod
    def enhance(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhances the input image tensor.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor [B, C, H, W] in [0, 1].
            
        Returns:
            torch.Tensor: Enhanced image tensor [B, C, H, W] in [0, 1].
        """
        pass
