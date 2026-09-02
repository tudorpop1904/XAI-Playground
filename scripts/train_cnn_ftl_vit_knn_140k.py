"""
scripts/train_cnn_ftl_vit_knn_140k.py
=======================================
Antrenament secvențial complet pe dataset-ul 140k:
  1. CNN (vanilla, 3 canale RGB)
  2. CNN + FTL (Full Transfer Learning channels: RGB + FFT + LBP + Sobel = 6 canale)
  3. ViT (Vision Transformer Tiny, 224x224)
  4. KNN (k-Nearest Neighbors cu backbone ResNet-18)

Fiecare etapa se logheaza atat in stdout cat si in training_140k.log (append).
La finalizare, afiseaza un tabel sumar cu val_acc si durata fiecarui stage.

Utilizare (in tmux):
--------------------
  # Sesiune noua tmux:
  tmux new -s training

  # Rulare cu valorile implicite (recomandate pentru 140k):
  python -u scripts/train_cnn_ftl_vit_knn_140k.py

  # Rulare cu parametri customizati:
  python -u scripts/train_cnn_ftl_vit_knn_140k.py \
      --epochs 15 \
      --batch-size 64 \
      --knn-k 7

  # Detasare fara a intrerupe antrenamentul:  Ctrl+B, then D
  # Re-atasare:  tmux attach -t training
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

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.detectors.cnn import CNNDetector
from core.detectors.vit import ViTDetector
from core.detectors.knn import KNNDetector
from core.utils.custom_dataset import FileListDataset
from core.utils.paths import MODELS_DIR, DATASETS_DIR

LOG_FILE = BASE_DIR / "training_140k.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("140k-Pipeline")


def get_dataset_index(dataset_slug: str) -> Path:
    dataset_name = dataset_slug.split("/")[-1]
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = DATASETS_DIR / f"{dataset_name}_index.json"

    if index_path.exists():
        logger.info(f"Index gasit la: {index_path}")
        return index_path

    available = list(DATASETS_DIR.glob("*_index.json"))
    if available:
        logger.warning(f"Index {index_path.name} negasit. Fallback: {available[0].name}")
        return available[0]

    logger.info(f"Descarca '{dataset_slug}' via kagglehub...")
    import kagglehub
    dpath = Path(kagglehub.dataset_download(dataset_slug, force_download=False))
    logger.info(f"Dataset descarcat la: {dpath}")

    real_images = (
        list(dpath.rglob("*real*/*.jpg")) + list(dpath.rglob("*real*/*.png"))
        + list(dpath.rglob("real/*.jpg")) + list(dpath.rglob("real/*.png"))
    )
    fake_images = (
        list(dpath.rglob("*fake*/*.jpg")) + list(dpath.rglob("*fake*/*.png"))
        + list(dpath.rglob("*ai*/*.jpg"))
        + list(dpath.rglob("fake/*.jpg")) + list(dpath.rglob("fake/*.png"))
    )

    if not real_images or not fake_images:
        logger.warning("Split real/fake negasit prin path. Se foloseste split 50/50.")
        all_imgs = list(dpath.rglob("*.jpg")) + list(dpath.rglob("*.png"))
        half = len(all_imgs) // 2
        real_images, fake_images = all_imgs[:half], all_imgs[half:]

    index_data = {
        "real": [str(p.resolve()) for p in real_images],
        "fake": [str(p.resolve()) for p in fake_images],
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f)

    logger.info(f"Index generat: {index_path} ({len(real_images)} Real, {len(fake_images)} Fake)")
    return index_path


def build_loaders(index_path: Path, img_size: int, batch_size: int, num_workers: int):
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = FileListDataset(index_data, transform=transform)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = dict(num_workers=num_workers, pin_memory=(device == "cuda"),
              persistent_workers=(num_workers > 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)

    logger.info(f"  Total: {total} | Train: {train_size} | Val: {val_size} | {img_size}x{img_size}")
    return train_loader, val_loader


def save_model(model, name: str, history: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pth"
    torch.save(model, path)
    with open(MODELS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"  Checkpoint salvat: {path}")
    return path


def banner(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def train_cnn(args, index_path: Path) -> dict:
    banner("STAGE 1/4 — CNN vanilla (RGB 128x128)")
    train_loader, val_loader = build_loaders(index_path, 128, args.batch_size, args.num_workers)
    model = CNNDetector(num_classes=2, add_fft=False, add_lbp=False, add_sobel=False)
    model.name = args.cnn_name
    logger.info(f"  {args.epochs} epoci, lr={args.cnn_lr}")
    t0 = time.time()
    history = model.train_model(train_loader=train_loader, val_loader=val_loader,
                                epochs=args.epochs, learning_rate=args.cnn_lr)
    elapsed = (time.time() - t0) / 60
    val_acc = history["val_acc"][-1]
    logger.info(f"  [CNN] Val Acc: {val_acc:.4%} | {elapsed:.2f} min")
    save_model(model, args.cnn_name, history)
    return {"model": "CNN", "val_acc": val_acc, "elapsed_min": elapsed}


def train_cnn_ftl(args, index_path: Path) -> dict:
    banner("STAGE 2/4 — CNN + FTL (6 canale: RGB + FFT + LBP + Sobel, 128x128)")
    train_loader, val_loader = build_loaders(index_path, 128, args.batch_size, args.num_workers)
    model = CNNDetector(num_classes=2, add_fft=True, add_lbp=True, add_sobel=True)
    model.name = args.ftl_name
    ch = getattr(model, "input_channels", "?")
    logger.info(f"  Canale input: {ch} | {args.epochs} epoci, lr={args.cnn_lr}")
    t0 = time.time()
    history = model.train_model(train_loader=train_loader, val_loader=val_loader,
                                epochs=args.epochs, learning_rate=args.cnn_lr)
    elapsed = (time.time() - t0) / 60
    val_acc = history["val_acc"][-1]
    history["specs"] = {"add_fft": True, "add_lbp": True, "add_sobel": True,
                        "input_channels": ch}
    logger.info(f"  [CNN+FTL] Val Acc: {val_acc:.4%} | {elapsed:.2f} min")
    save_model(model, args.ftl_name, history)
    return {"model": "CNN+FTL", "val_acc": val_acc, "elapsed_min": elapsed}


def train_vit(args, index_path: Path) -> dict:
    banner("STAGE 3/4 — ViT Tiny (224x224, AdamW + CosineAnnealing)")
    train_loader, val_loader = build_loaders(index_path, 224, args.batch_size, args.num_workers)
    model = ViTDetector(num_classes=2, img_size=224, patch_size=16,
                        embed_dim=192, depth=12, num_heads=3)
    model.name = args.vit_name
    logger.info(f"  ViT-Tiny (~5.7M params) | {args.epochs} epoci, lr={args.vit_lr}")
    t0 = time.time()
    history = model.train_model(train_loader=train_loader, val_loader=val_loader,
                                epochs=args.epochs, lr=args.vit_lr)
    elapsed = (time.time() - t0) / 60
    val_acc = history["val_acc"][-1]
    logger.info(f"  [ViT] Val Acc: {val_acc:.4%} | {elapsed:.2f} min")
    save_model(model, args.vit_name, history)
    return {"model": "ViT", "val_acc": val_acc, "elapsed_min": elapsed}


def train_knn(args, index_path: Path) -> dict:
    banner(f"STAGE 4/4 — k-NN (k={args.knn_k}, {args.knn_metric}, backbone={args.knn_backbone})")
    train_loader, val_loader = build_loaders(index_path, 224, args.batch_size, args.num_workers)
    model = KNNDetector(k=args.knn_k, metric=args.knn_metric, backbone=args.knn_backbone)
    model.name = args.knn_name
    logger.info("  Extragere features si fitting k-NN...")
    t0 = time.time()
    history = model.train_model(train_loader=train_loader, val_loader=val_loader)
    elapsed = time.time() - t0
    val_acc = history["val_acc"][0]
    logger.info(f"  [k-NN] Val Acc: {val_acc:.4%} | {elapsed:.2f} sec")
    save_model(model, args.knn_name, history)
    return {"model": f"KNN(k={args.knn_k})", "val_acc": val_acc, "elapsed_min": elapsed / 60}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline CNN + CNN+FTL + ViT + KNN pe 140k",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparametri principali
    p.add_argument("--epochs",      type=int,   default=10,    help="Epoci pentru CNN si ViT")
    p.add_argument("--batch-size",  type=int,   default=64,    help="Batch size")
    p.add_argument("--knn-k",       type=int,   default=5,     help="k pentru k-NN")
    # Avansati
    p.add_argument("--cnn-lr",       type=float, default=1e-3)
    p.add_argument("--vit-lr",       type=float, default=3e-4)
    p.add_argument("--knn-metric",   type=str,   default="cosine",   choices=["cosine", "euclidean"])
    p.add_argument("--knn-backbone", type=str,   default="resnet18", choices=["resnet18", "ftl_cnn", "vit"])
    p.add_argument("--num-workers",  type=int,   default=4)
    # Nume modele
    p.add_argument("--cnn-name",  type=str, default="CNN_140k_1")
    p.add_argument("--ftl-name",  type=str, default="CNN_FTL_140k_1")
    p.add_argument("--vit-name",  type=str, default="ViT_140k_1")
    p.add_argument("--knn-name",  type=str, default="KNN_140k_1")
    # Dataset
    p.add_argument("--dataset-slug", type=str, default="140k-real-and-fake-faces/dataset")
    # Skip flags
    p.add_argument("--no-cnn", action="store_true", help="Omite CNN vanilla")
    p.add_argument("--no-ftl", action="store_true", help="Omite CNN+FTL")
    p.add_argument("--no-vit", action="store_true", help="Omite ViT")
    p.add_argument("--no-knn", action="store_true", help="Omite k-NN")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_pipeline = time.time()

    banner("PIPELINE 140k — CNN + CNN+FTL + ViT + k-NN")
    logger.info(f"  Device          : {device.upper()}")
    if device == "cuda":
        logger.info(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  VRAM totala     : {vram:.2f} GB")
    logger.info(f"  Epoci (CNN/ViT) : {args.epochs}")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  k (KNN)         : {args.knn_k}")
    logger.info(f"  Dataset slug    : {args.dataset_slug}")
    logger.info(f"  Log file        : {LOG_FILE}")
    active = [s for s, skip in [("CNN", args.no_cnn), ("CNN+FTL", args.no_ftl),
                                  ("ViT", args.no_vit), ("KNN", args.no_knn)] if not skip]
    logger.info(f"  Stage-uri active: {', '.join(active)}")

    index_path = get_dataset_index(args.dataset_slug)
    results = []

    if not args.no_cnn:
        try:   results.append(train_cnn(args, index_path))
        except Exception as e: logger.error(f"[STAGE CNN] ESUAT: {e}", exc_info=True)

    if not args.no_ftl:
        try:   results.append(train_cnn_ftl(args, index_path))
        except Exception as e: logger.error(f"[STAGE CNN+FTL] ESUAT: {e}", exc_info=True)

    if not args.no_vit:
        try:   results.append(train_vit(args, index_path))
        except Exception as e: logger.error(f"[STAGE ViT] ESUAT: {e}", exc_info=True)

    if not args.no_knn:
        try:   results.append(train_knn(args, index_path))
        except Exception as e: logger.error(f"[STAGE KNN] ESUAT: {e}", exc_info=True)

    total_min = (time.time() - t_pipeline) / 60
    banner("SUMAR FINAL")
    logger.info(f"  {'Model':<20} {'Val Acc':>10} {'Durata':>12}")
    logger.info(f"  {'-'*20} {'-'*10} {'-'*12}")
    for r in results:
        logger.info(f"  {r['model']:<20} {r['val_acc']:>10.4%} {r['elapsed_min']:>10.2f} min")
    logger.info(f"  {'='*44}")
    logger.info(f"  {'TOTAL':<20} {'':>10} {total_min:>10.2f} min")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
