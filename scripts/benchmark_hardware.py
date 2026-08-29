"""
scripts/benchmark_hardware.py
==============================
Automated hardware & latency benchmarking harness.

Measures across both the local (edge) and GCP cloud environments:
  - CUDA / GPU availability & device name
  - System RAM
  - VRAM usage
  - Inference latency per detector (CNN, ViT, KNN)
  - XAI explanation latency per explainer (Grad-CAM, Saliency, Occlusion, PMI, Sobol)
  - XAI metric values (Sparsity, Faithfulness, Stability)

Output:
  - Prints a formatted table to stdout
  - Saves a structured JSON report to storage/benchmark_results.json
  
Usage:
  docker compose exec api python scripts/benchmark_hardware.py
"""

import json
import os
import sys
import time
import tempfile
import platform
from pathlib import Path

import torch
import psutil

# ── Ensure project root is in sys.path ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detectors.base import AbstractBaseDetector
from core.explainers.base import BaseExplainer
from core.utils.metrics import compute_sparsity, compute_faithfulness, compute_stability

# ── Constants ─────────────────────────────────────────────────────
STORAGE = Path(__file__).resolve().parent.parent / "storage"
OUTPUT_PATH = STORAGE / "benchmark_results.json"

DETECTORS = ["CNN_Model", "ViT_Model", "KNN_Model"]
EXPLAINERS = [
    ("grad_cam",          {}),
    ("vanilla_saliency",  {}),
    ("occlusion",         {"grid_rows": 4, "grid_cols": 4}),
    ("pmi",               {"grid_rows": 4, "grid_cols": 4}),
    ("sobol",             {"grid_rows": 4, "grid_cols": 4, "n_samples": 64}),
]

SEPARATOR = "=" * 72


# ── Utilities ─────────────────────────────────────────────────────

def get_system_info() -> dict:
    """Collect platform, CPU, RAM, and GPU information."""
    info = {
        "platform":        platform.system(),
        "python_version":  platform.python_version(),
        "cpu":             platform.processor() or platform.machine(),
        "ram_total_gb":    round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cuda_available":  torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"]      = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"]   = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        info["cuda_version"]  = torch.version.cuda
    else:
        info["gpu_name"]      = "N/A (CPU-only)"
        info["gpu_vram_gb"]   = 0.0
        info["cuda_version"]  = "N/A"
    return info


def create_dummy_tensor() -> Path:
    """Save a 224×224 dummy RGB image tensor to a temp file and return its path."""
    tensor = torch.randn(1, 3, 224, 224)
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", dir=STORAGE / "images", delete=False)
    torch.save(tensor, tmp.name)
    return Path(tmp.name)


def measure_vram_delta(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) and return (result, peak_vram_mb).
    Falls back to 0.0 if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        vram_mb = round((peak - before) / (1024 ** 2), 2)
    else:
        vram_mb = 0.0

    return result, elapsed, vram_mb


# ── Main Benchmark ────────────────────────────────────────────────

def run_benchmarks():
    (STORAGE / "images").mkdir(parents=True, exist_ok=True)

    print(SEPARATOR)
    print("  XAI Playground — Hardware & Latency Benchmark Harness")
    print(SEPARATOR)

    system_info = get_system_info()
    print(f"  Platform  : {system_info['platform']} / Python {system_info['python_version']}")
    print(f"  CPU       : {system_info['cpu']}")
    print(f"  RAM       : {system_info['ram_total_gb']} GB")
    print(f"  GPU       : {system_info['gpu_name']}")
    print(f"  VRAM      : {system_info['gpu_vram_gb']} GB")
    print(f"  CUDA      : {system_info['cuda_version']}")
    print(SEPARATOR)

    dummy_path = create_dummy_tensor()
    dummy_tensor = torch.load(dummy_path, weights_only=False)

    results = {
        "system": system_info,
        "detectors": {},
        "explainers": {},
    }

    # ── Detector Benchmarks ──────────────────────────────────────
    print("\n[1/2] DETECTOR INFERENCE LATENCY")
    print(f"  {'Detector':<20} {'Latency (s)':>12} {'Peak VRAM (MB)':>16} {'Prediction'}")
    print(f"  {'-'*20} {'-'*12} {'-'*16} {'-'*15}")

    # Use CNN as the primary detector for XAI benchmarks (it's most universally available)
    cnn_model = None

    for det_name in DETECTORS:
        try:
            model = AbstractBaseDetector.get_by_name(det_name)
            detection, elapsed, vram = measure_vram_delta(model.predict, str(dummy_path))
            pred = "AI" if detection.ai_deepfake else "Real"
            print(f"  {det_name:<20} {elapsed:>12.4f} {vram:>16.2f} {pred}")
            results["detectors"][det_name] = {
                "latency_s":     round(elapsed, 4),
                "peak_vram_mb":  vram,
                "prediction":    pred,
                "confidence":    round(detection.confidence, 4),
            }
            if det_name == "CNN_Model":
                cnn_model = model
        except Exception as e:
            print(f"  {det_name:<20} {'ERROR':>12} {'N/A':>16}  {str(e)[:40]}")
            results["detectors"][det_name] = {"error": str(e)}

    # ── Explainer Benchmarks ─────────────────────────────────────
    print("\n[2/2] XAI EXPLAINER LATENCY & METRICS (using CNN_Model)")
    print(f"  {'Explainer':<22} {'Latency (s)':>12} {'VRAM (MB)':>10} {'Sparsity':>10} {'Faithful':>10} {'Stable':>8}")
    print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    if cnn_model is None:
        print("  [SKIP] CNN_Model failed to load; skipping XAI benchmarks.")
    else:
        detection = cnn_model.predict(str(dummy_path))

        for expl_name, expl_kwargs in EXPLAINERS:
            try:
                explainer = BaseExplainer.get_explainer(expl_name, **expl_kwargs)
                explanation, elapsed, vram = measure_vram_delta(
                    explainer.explain,
                    cnn_model,
                    str(dummy_path),
                    int(detection.ai_deepfake)
                )
                heatmap = explanation.returned_obj

                # Compute XAI quality metrics
                sparsity    = compute_sparsity(heatmap)
                faithfulness = compute_faithfulness(cnn_model, dummy_tensor, heatmap, int(detection.ai_deepfake))
                stability   = compute_stability(explainer, str(dummy_path), int(detection.ai_deepfake), cnn_model)

                print(f"  {expl_name:<22} {elapsed:>12.4f} {vram:>10.2f} {sparsity:>10.4f} {faithfulness:>10.4f} {stability:>8.4f}")
                results["explainers"][expl_name] = {
                    "latency_s":      round(elapsed, 4),
                    "peak_vram_mb":   vram,
                    "sparsity":       sparsity,
                    "faithfulness":   faithfulness,
                    "stability":      stability,
                }
            except Exception as e:
                print(f"  {expl_name:<22} {'ERROR':>12} {'N/A':>10}  {str(e)[:40]}")
                results["explainers"][expl_name] = {"error": str(e)}

    # ── Write JSON Report ─────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{SEPARATOR}")
    print(f"  Report saved to: {OUTPUT_PATH}")
    print(SEPARATOR)

    # Cleanup dummy tensor
    dummy_path.unlink(missing_ok=True)

    return results


if __name__ == "__main__":
    run_benchmarks()
