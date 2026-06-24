import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Setup import paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from net.st_gcn import Model
from net.body_part import PART_NAMES
from analysis.temporal_segmentation.motion_energy import resolve_data_path

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # 1. Load trained model
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
    else:
        print("No weights found at:", weights_path, "- running with random initialization.")
    
    model.eval()
    
    # 2. Load realistic sample 0
    data_path = resolve_data_path()
    data = np.load(data_path, mmap_mode='r')
    sample = torch.tensor(data[0:1]).float().to(device)  # (1, 3, 300, 25, 2)
    
    # 3. Forward pass up to body-part features
    with torch.no_grad():
        x, N, M = model._forward_backbone(sample)  # (N*M, 256, T', 25)
        part_x = model.body_part(x)  # (N*M, 256, T', 6)
        
    print("Real skeleton backbone output shape:", part_x.shape)
    
    # 4. Compute dynamic adjacency matrix (N*M, 6, 6)
    n_dyn, c_dyn, t_dyn, p_dyn = part_x.size()
    part_flat = part_x.permute(0, 2, 1, 3).contiguous().view(n_dyn, t_dyn * c_dyn, p_dyn)
    part_mean = part_flat.mean(dim=1, keepdim=True)
    part_centered = part_flat - part_mean
    part_std = part_centered.norm(dim=1, keepdim=True) + 1e-8
    part_norm = part_centered / part_std
    adj_dynamic = torch.bmm(part_norm.transpose(1, 2), part_norm)
    
    # Move to CPU for diagnostics
    adj_cpu = adj_dynamic.cpu().numpy()
    
    # 5. Print one complete adjacency matrix (Sample 0, Person 0)
    print("\n--- Complete Dynamic Adjacency Matrix (Sample 0, Person 0) ---")
    short_labels = ["H", "T", "LA", "RA", "LL", "RL"]
    print("    " + " ".join([f"{lbl:>6}" for lbl in short_labels]))
    for i in range(p_dyn):
        row_str = f"{short_labels[i]:<3}"
        for j in range(p_dyn):
            row_str += f" {adj_cpu[0, i, j]:6.3f}"
        print(row_str)
        
    # 6. Statistics for each sample
    print("\n--- Statistics for each sample/person in the batch ---")
    for b in range(n_dyn):
        s_mat = adj_cpu[b]
        s_mean = s_mat.mean()
        s_std = s_mat.std()
        s_max = s_mat.max()
        s_min = s_mat.min()
        print(f"Sample {b} (Person {b}): Mean={s_mean:.4f}, Std={s_std:.4f}, Max={s_max:.4f}, Min={s_min:.4f}")
        
    # 7. Compute off-diagonal statistics
    print("\n--- Off-diagonal Statistics ---")
    for b in range(n_dyn):
        s_mat = adj_cpu[b]
        mask = ~np.eye(p_dyn, dtype=bool)
        off_diag = s_mat[mask]
        od_mean = off_diag.mean()
        od_min = off_diag.min()
        od_max = off_diag.max()
        print(f"Sample {b} Off-Diagonal: Mean={od_mean:.4f}, Min={od_min:.4f}, Max={od_max:.4f}")
        
    # Find strongest and weakest off-diagonal pairs for Sample 0
    od_pairs = []
    for i in range(p_dyn):
        for j in range(i + 1, p_dyn):
            od_pairs.append(((PART_NAMES[i], PART_NAMES[j]), adj_cpu[0, i, j]))
    sorted_pairs = sorted(od_pairs, key=lambda x: x[1], reverse=True)
    
    print("\nSample 0 (Person 0) Off-Diagonal Rankings:")
    print("  Strongest:")
    for (p1, p2), val in sorted_pairs[:3]:
        print(f"    {p1} <-> {p2}: {val:.4f}")
    print("  Weakest:")
    for (p1, p2), val in reversed(sorted_pairs[-3:]):
        print(f"    {p1} <-> {p2}: {val:.4f}")
        
    # 8. Save Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(adj_cpu[0], cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(PART_NAMES)))
    ax.set_yticks(np.arange(len(PART_NAMES)))
    ax.set_xticklabels(PART_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(PART_NAMES)
    for i in range(p_dyn):
        for j in range(p_dyn):
            ax.text(j, i, f"{adj_cpu[0, i, j]:.2f}", ha="center", va="center", color="black" if abs(adj_cpu[0, i, j]) < 0.7 else "white")
    plt.colorbar(im)
    ax.set_title("Sample 0 Person 0 Dynamic Adjacency (Pearson)")
    plt.tight_layout()
    output_dir = os.path.join(REPO_ROOT, 'analysis', 'cross_body_interaction', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, 'dynamic_adjacency_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print("\nSaved heatmap to:", heatmap_path)

if __name__ == '__main__':
    main()
