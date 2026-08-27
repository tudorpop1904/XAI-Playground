import torch
from pathlib import Path
from PIL import Image
import io
import torchvision.transforms as transforms
from fastapi import APIRouter, UploadFile, File

from core.enhancers.super_resolution import SuperResolutionEnhancer

router = APIRouter(prefix="/api/v1")
device = "cuda" if torch.cuda.is_available() else "cpu"

@router.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Endpoint to enhance an uploaded image using the Super Resolution model.
    """
    
    # Read the image file and convert to tensor
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Transform to [1, C, H, W] normalized tensor
    transform = transforms.ToTensor()
    image_tensor = transform(img).unsqueeze(0)
    
    # Enhance the image
    enhancer = SuperResolutionEnhancer()
    enhanced_tensor = enhancer.enhance(image_tensor)

    # Save the enhanced tensor and a visual JPG back to the storage
    from api.main import IMAGES
    filename_stem = Path(file.filename).stem
    
    tensor_path = IMAGES / f"{filename_stem}_enhanced.pt"
    torch.save(enhanced_tensor, tensor_path)
    
    img_path = IMAGES / f"{filename_stem}_enhanced.jpg"
    from torchvision.utils import save_image
    save_image(enhanced_tensor, img_path)

    return {
        "tensor_url": f"/storage/images/{tensor_path.name}", 
        "image_url": f"/storage/images/{img_path.name}"
    }