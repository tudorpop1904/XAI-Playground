"""
scripts/train_vit_knn_pipeline.py
==================================
Automated end-to-end training pipeline:
1. Trains ViT (Vision Transformer) detector with AdamW + CosineAnnealing.
2. Upon successful completion, immediately trains k-NN detector with representation transfer.
3. Automatically persists checkpoints (.pth) and training metadata (.json) to storage/models/.

Usage:
------
python -u scripts/train_vit_knn_pipeline.py --dataset-slug birdy654/cifake-real-and-ai-generated-synthetic-images --vit-epochs 10 --batch-size 32
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.detectors.vit import ViTDetector
from core.detectors.knn import KNNDetector
from core.utils.custom_dataset import FileListDataset
from core.utils.paths import MODELS_DIR, DATASETS_DIR

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "training_run.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("TrainingPipeline")


def get_dataset_index(dataset_slug: str) -> Path:
    """Finds or validates the index JSON for the given dataset slug."""
    dataset_name = dataset_slug.split("/")[-1]
    index_path = DATASETS_DIR / f"{dataset_name}_index.json"
    
    if not index_path.exists():
        # Fallback search for any available index
        available = list(DATASETS_DIR.glob("*_index.json"))
        if available:
            logger.warning(f"Requested index {index_path.name} not found. Using available: {available[0].name}")
            return available[0]
        raise FileNotFoundError(
            f"Dataset index not found at {index_path}. "
            "Please download/prepare the dataset via UI or API first."
        )
    return index_path


def main():
    parser = argparse.ArgumentParser(description="Automated ViT + k-NN Training Pipeline")
    parser.add_argument("--dataset-slug", type=str, default="birdy654/cifake-real-and-ai-generated-synthetic-images")
    parser.add_argument("--vit-epochs", type=int, default=10, help="Epochs for ViT training")
    parser.add_argument("--vit-lr", type=float, default=3e-4, help="Learning rate for ViT (AdamW)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--vit-name", type=str, default="ViT_Model_Auto_1")
    parser.add_argument("--knn-name", type=str, default="KNN_Model_Auto_1")
    parser.add_argument("--knn-k", type=int, default=5, help="Number of neighbors for k-NN")
    parser.add_argument("--knn-metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--knn-backbone", type=str, default="resnet18", choices=["resnet18", "ftl_cnn", "vit"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 70)
    logger.info(" STARTING AUTOMATED SEQUENTIAL TRAINING PIPELINE")
    logger.info(f" Compute Device  : {device.upper()}")
    if device == "cuda":
        logger.info(f" GPU Name        : {torch.cuda.get_device_name(0)}")
        logger.info(f" Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    logger.info(f" Dataset Target  : {args.dataset_slug}")
    logger.info(f" ViT Epochs      : {args.vit_epochs}")
    logger.info(f" Batch Size      : {args.batch_size}")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Dataset Loading & DataLoader Setup
    # -------------------------------------------------------------------------
    index_path = get_dataset_index(args.dataset_slug)
    logger.info(f"Loading dataset index from: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    # Transform: resize to 224x224 for ViT & Pretrained ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FileListDataset(index_data, transform=transform)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    logger.info(f"Total Samples: {total_samples} (Train: {train_size}, Validation: {val_size})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. STAGE 1: Train ViT (Vision Transformer)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f" STAGE 1: TRAINING VISION TRANSFORMER ({args.vit_name})")
    logger.info("=" * 70)

    t0_vit = time.time()
    vit_model = ViTDetector(
        num_classes=2,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
    )
    vit_model.name = args.vit_name

    logger.info(f"Instantiated ViT-Tiny architecture (~5.7M parameters). Training for {args.vit_epochs} epochs...")
    vit_history = vit_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.vit_epochs,
        lr=args.vit_lr,
    )

    vit_time = time.time() - t0_vit
    logger.info(f"[ViT] Training completed in {vit_time / 60:.2f} minutes.")
    logger.info(f"[ViT] Final Train Acc: {vit_history['train_acc'][-1]:.4f} | Final Val Acc: {vit_history['val_acc'][-1]:.4f}")

    # Save ViT
    vit_path = MODELS_DIR / f"{args.vit_name}.pth"
    torch.save(vit_model, vit_path)
    with open(MODELS_DIR / f"{args.vit_name}.json", "w", encoding="utf-8") as f:
        json.dump(vit_history, f, indent=2)
    logger.info(f"[ViT] Checkpoint saved successfully to {vit_path}")

    # -------------------------------------------------------------------------
    # 3. STAGE 2: Train k-NN (Representation Transfer)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f" STAGE 2: FITTING k-NN DETECTOR ({args.knn_name})")
    logger.info(f" Backbone: {args.knn_backbone} | k={args.knn_k} | Metric={args.knn_metric}")
    logger.info("=" * 70)

    t0_knn = time.time()
    knn_model = KNNDetector(
        k=args.knn_k,
        metric=args.knn_metric,
        backbone=args.knn_backbone,
    )
    knn_model.name = args.knn_name

    knn_history = knn_model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    knn_time = time.time() - t0_knn
    logger.info(f"[k-NN] Feature extraction and fitting completed in {knn_time:.2f} seconds.")
    logger.info(f"[k-NN] Leave-One-Out Train Acc: {knn_history['train_acc'][0]:.4f} | Val Acc: {knn_history['val_acc'][0]:.4f}")

    # Save k-NN
    knn_path = MODELS_DIR / f"{args.knn_name}.pth"
    torch.save(knn_model, knn_path)
    with open(MODELS_DIR / f"{args.knn_name}.json", "w", encoding="utf-8") as f:
        json.dump(knn_history, f, indent=2)
    logger.info(f"[k-NN] Checkpoint saved successfully to {knn_path}")

    # -------------------------------------------------------------------------
    # 4. Pipeline Summary
    # -------------------------------------------------------------------------
    total_time = (time.time() - t0_vit) / 60
    logger.info("\n" + "=" * 70)
    logger.info(" PIPELINE FINISHED SUCCESSFULLY!")
    logger.info(f" Total Runtime    : {total_time:.2f} minutes")
    logger.info(f" ViT Model Saved  : {vit_path.name} (Val Acc: {vit_history['val_acc'][-1]:.4%})")
    logger.info(f" k-NN Model Saved : {knn_path.name} (Val Acc: {knn_history['val_acc'][0]:.4%})")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
