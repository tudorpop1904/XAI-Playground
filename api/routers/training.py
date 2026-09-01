from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

from core.detectors.base import AbstractBaseDetector
from core.detectors.cnn import CNNDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

router = APIRouter(prefix="/api/v1")

class TrainRequest(BaseModel):
    model_name: str
    dataset_slug: str
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    freeze_backbone: bool = False
    model_type: str = "CNN"
    add_fft: bool = False
    add_lbp: bool = False
    add_sobel: bool = False

@router.post("/models/train")
def train_model(req: TrainRequest):
    """
    Train a model on a pre-prepared dataset subset with custom forensic channels.
    """
    dataset_name = req.dataset_slug.split("/")[-1]
    index_path = Path("storage/datasets") / f"{dataset_name}_index.json"
    
    if not index_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset index {dataset_name} not found. Call /prepare first.")

    # Prepare DataLoaders
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    try:
        import json
        from core.utils.custom_dataset import FileListDataset
        
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
            
        dataset = FileListDataset(index_data, transform=transform)
        # 80-20 split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=req.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=req.batch_size, shuffle=False)
        
        # Instantiate model if it's a new one, or load it
        try:
            model = AbstractBaseDetector.get_by_name(req.model_name)
        except Exception:
            m_type = (req.model_type or req.model_name.split("_")[0]).upper()
            if m_type == "CNN":
                model = CNNDetector(
                    num_classes=2,
                    add_fft=req.add_fft,
                    add_lbp=req.add_lbp,
                    add_sobel=req.add_sobel,
                )
            elif m_type == "VIT":
                from core.detectors.vit import ViTDetector
                model = ViTDetector()
            elif m_type == "KNN":
                from core.detectors.knn import KNNDetector
                model = KNNDetector()
            elif m_type == "KMC":
                from core.detectors.kmc import KMCDetector
                model = KMCDetector()
            else:
                model = CNNDetector(
                    num_classes=2,
                    add_fft=req.add_fft,
                    add_lbp=req.add_lbp,
                    add_sobel=req.add_sobel,
                )
            model.name = req.model_name
            
        # Train (pass freeze_backbone if supported)
        import inspect
        sig = inspect.signature(model.train_model)
        train_kwargs = {"epochs": req.epochs, "learning_rate": req.learning_rate}
        if "freeze_backbone" in sig.parameters:
            train_kwargs["freeze_backbone"] = req.freeze_backbone
            
        history = model.train_model(train_loader, val_loader, **train_kwargs)
        
        # Save model
        MODELS_DIR = Path("storage/models")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model, MODELS_DIR / f"{req.model_name}.pth")
        
        # Save training history metadata and channel configuration
        if isinstance(history, dict):
            history["specs"] = {
                "model_type": getattr(model, "__class__", type(model)).__name__,
                "add_fft": getattr(model, "add_fft", False),
                "add_lbp": getattr(model, "add_lbp", False),
                "add_sobel": getattr(model, "add_sobel", False),
                "input_channels": getattr(model, "input_channels", 3),
            }

        with open(MODELS_DIR / f"{req.model_name}.json", "w") as f:
            json.dump(history, f)
            
        # Update cache
        AbstractBaseDetector._cache[req.model_name] = model
        
        return {
            "status": "success",
            "history": history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
def list_models():
    """
    Returns a list of trained models available in storage, along with their metrics.
    """
    import json
    MODELS_DIR = Path("storage/models")
    if not MODELS_DIR.exists():
        return []
        
    models = []
    for pth_file in MODELS_DIR.glob("*.pth"):
        model_name = pth_file.stem
        
        # Try to load history
        history = {}
        json_file = MODELS_DIR / f"{model_name}.json"
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    history = json.load(f)
            except Exception:
                pass
                
        models.append({
            "name": model_name,
            "history": history
        })
        
    return models
