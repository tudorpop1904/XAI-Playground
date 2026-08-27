import os
import cv2
import torch
import numpy as np
from core.enhancers.base import BaseEnhancer

class SuperResolutionEnhancer(BaseEnhancer):
    """
    AI Image Enhancer using OpenCV's DNN Super Resolution.
    Uses the EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) model.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str = "storage/models/EDSR_x4.pb", scale: int = 4):
        if getattr(self, '_initialized', False):
            return
            
        self.scale = scale
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please download it.")
            
        # Initialize OpenCV super resolution object
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel("edsr", self.scale)
        self._initialized = True

    def enhance(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Upscales and enhances the input image.
        Expects a PyTorch tensor [1, C, H, W] normalized to [0, 1].
        """
        device = image_tensor.device
        
        # Convert PyTorch tensor to OpenCV format (H, W, C) in BGR and [0, 255]
        # We assume batch size 1 for simplicity here
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # PyTorch uses RGB, OpenCV uses BGR
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Run super resolution
        enhanced_bgr = self.sr.upsample(img_bgr)
        
        # Convert back to PyTorch format (RGB, [0, 1], [1, C, H, W])
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        enhanced_tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float() / 255.0
        enhanced_tensor = enhanced_tensor.unsqueeze(0).to(device)
        
        return enhanced_tensor
