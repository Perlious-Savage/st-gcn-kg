import os
import sys
import numpy as np
import torch

# Setup import paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from net.st_gcn import Model
from net.body_part import PART_NAMES
from analysis.temporal_segmentation.motion_energy import resolve_data_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 1. Load model
    model = Model(
        in_channels=3,
        num_class=120,
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
        edge_importance_weighting=True,
        use_kg=True
    ).to(device)
    
    weights_path = os.path.join(REPO_ROOT, 'work_dir', 'recognition', 'ntu-xsub', 'ST_GCN', 'epoch10_model.pt')
    if os.path.isfile(weights_path):
        print("Loading weights from:", weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    
    model.eval()
    
    # 2. Load validation sample 0 (batch size = 1, persons = 2)
    data_path = resolve_data_path()
    data = np.load(data_path, mmap_mode='r')
    sample = torch.tensor(data[0:1]).float().to(device)  # (1, 3, 300, 25, 2)
    
    # 3. Forward pass to extract part_x
    with torch.no_grad():
        x, N, M = model._forward_backbone(sample)
        part_x = model.body_part(x)  # (N*M, 256, T', 6)
        
        # Compute dynamic adjacency matrix
        n_dyn, c_dyn, t_dyn, p_dyn = part_x.size()
        part_flat = part_x.permute(0, 2, 1, 3).contiguous().view(n_dyn, t_dyn * c_dyn, p_dyn)
        part_mean = part_flat.mean(dim=1, keepdim=True)
        part_centered = part_flat - part_mean
        part_std = part_centered.norm(dim=1, keepdim=True) + 1e-8
        part_norm = part_centered / part_std
        adj_dynamic = torch.bmm(part_norm.transpose(1, 2), part_norm)
        
        # 4. Message passing under Static KG
        kg_output_static = model.kg_gnn(part_x, adj_dynamic=None)
        
        # 5. Message passing under Dynamic KG
        kg_output_dynamic = model.kg_gnn(part_x, adj_dynamic=adj_dynamic)
        
    # Move to CPU for metrics
    static_np = kg_output_static.cpu().numpy()
    dynamic_np = kg_output_dynamic.cpu().numpy()
    
    # 6. Compute differences
    mean_abs_diff = np.mean(np.abs(static_np - dynamic_np))
    max_abs_diff = np.max(np.abs(static_np - dynamic_np))
    
    # Percentage of elements changed (using a tiny threshold to account for precision)
    num_changed = np.sum(np.abs(static_np - dynamic_np) > 1e-5)
    pct_changed = (num_changed / static_np.size) * 100
    
    print("\n--- Forward Pass Comparison ---")
    print(f"Mean Absolute Difference: {mean_abs_diff:.6f}")
    print(f"Max Absolute Difference : {max_abs_diff:.6f}")
    print(f"Percentage of elements changed (>1e-5): {pct_changed:.2f}%")
    
    # 7. Print dynamic adjacency matrix passed to GNN for Sample 0 Person 0
    adj_cpu = adj_dynamic.cpu().numpy()
    print("\n--- Dynamic Adjacency Matrix passed to GNN (Sample 0, Person 0) ---")
    short_labels = ["H", "T", "LA", "RA", "LL", "RL"]
    print("    " + " ".join([f"{lbl:>6}" for lbl in short_labels]))
    for i in range(p_dyn):
        row_str = f"{short_labels[i]:<3}"
        for j in range(p_dyn):
            row_str += f" {adj_cpu[0, i, j]:6.3f}"
        print(row_str)

if __name__ == '__main__':
    main()
