
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from api.schemas.requests import TrainRequest
from core.detectors.base import AbstractBaseDetector
from core.utils.paths import DATASETS_DIR, MODELS_DIR


def train(req: TrainRequest):
    dataset_name = req.dataset_slug.split("/")[-1]
    index_path = DATASETS_DIR / f"{dataset_name}_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index {dataset_name} not found. Call /prepare first.")

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
                from core.detectors.cnn import CNNDetector
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
                model = KNNDetector(
                    k=req.knn_k,
                    metric=req.knn_metric,
                    backbone=req.knn_backbone,
                )
            else:
                from core.detectors.cnn import CNNDetector
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
        train_kwargs = {}
        if "freeze_backbone" in sig.parameters:
            train_kwargs["freeze_backbone"] = req.freeze_backbone
        if "lr" in sig.parameters:
            train_kwargs["lr"] = req.learning_rate
        if "epochs" in sig.parameters:
            train_kwargs["epochs"] = req.epochs

        history = model.train_model(train_loader, val_loader, **train_kwargs)
        
        # Save model
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
        
        return {
            "status": "success",
            "history": history
        }

    except Exception as e:
        raise ValueError(str(e))

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()

    parser.add_argument("--payload", required=True, help="TrainRequest as JSON")
    
    args = parser.parse_args()

    req = TrainRequest.model_validate_json(args.payload)

    try:
        train(req)
        sys.exit(0)
    except Exception as e:
        print(f"[train_worker] FAILED: {e}", file=sys.stderr)
        sys.exit(1)