import torch
from core.detectors.cnn import CNNDetector
from core.utils.metrics import compute_faithfulness, compute_stability
import matplotlib.pyplot as plt

def debug_metrics():
    model = CNNDetector()
    model.eval()
    
    # create a dummy image (1, 3, 128, 128)
    image = torch.rand(1, 3, 128, 128)
    
    # create a dummy heatmap (128, 128)
    heatmap = torch.rand(128, 128)
    
    # 1. Test Faithfulness
    f_val = compute_faithfulness(model, image, heatmap, target_class=0)
    print(f"Faithfulness on random heatmap: {f_val}")
    
    # 2. Check mask logic
    h_flat = heatmap.flatten()
    k = max(1, int(0.2 * h_flat.numel()))
    threshold_val = torch.topk(h_flat, k).values[-1]
    mask = (heatmap < threshold_val).float()
    print(f"Mask mean (should be ~0.8): {mask.mean().item()}")
    
    # 3. What if heatmap is mostly zero?
    sparse_heatmap = torch.zeros(128, 128)
    sparse_heatmap[0:10, 0:10] = 1.0  # tiny blob
    
    h_flat_s = sparse_heatmap.flatten()
    k_s = max(1, int(0.4 * h_flat_s.numel()))
    
    top_indices = torch.topk(h_flat_s, k_s).indices
    mask_flat = torch.ones_like(h_flat_s)
    mask_flat[top_indices] = 0.0
    mask_s = mask_flat.view(sparse_heatmap.shape)
    
    print(f"Sparse Mask mean (should be ~0.6): {mask_s.mean().item()}")
    
if __name__ == "__main__":
    debug_metrics()
