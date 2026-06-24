import os
import sys
import numpy as np
import torch

# Setup import paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from net.body_part import BodyPartAggregator, PART_NAMES
from analysis.temporal_segmentation.motion_energy import resolve_data_path

def main():
    # 1. Load data & compute motion magnitudes (similar to body_part_velocity.py)
    data_path = resolve_data_path()
    data = np.load(data_path, mmap_mode='r')
    sample_idx = 0
    sample = np.array(data[sample_idx])  # (C, T, V, M)
    
    # Person-1 only
    skeleton_ctv = sample[:, :, :, 0]  # (3, T, V)
    
    # Aggregation
    x_in = torch.tensor(skeleton_ctv).unsqueeze(0).float()
    aggregator = BodyPartAggregator()
    with torch.no_grad():
        y_out = aggregator(x_in)  # (1, C, T, P)
        
    part_features = y_out.squeeze(0).permute(1, 2, 0).numpy()  # (T, P, C)
    
    # Velocities
    T, P, C = part_features.shape
    velocity_part = np.zeros_like(part_features)
    velocity_part[1:] = part_features[1:] - part_features[:-1]
    
    # Motion magnitudes
    speed_part = np.linalg.norm(velocity_part, axis=-1)  # (T, P)
    
    # 2. Pearson correlation matrix (6x6)
    # np.corrcoef expects (variables, observations) if rowvar=True (default),
    # or (observations, variables) if rowvar=False.
    # speed_part shape is (300, 6), i.e., (observations, variables).
    corr_matrix = np.corrcoef(speed_part, rowvar=False)
    
    # 3. Print matrix with labels
    short_labels = ["H", "T", "LA", "RA", "LL", "RL"]
    
    print("\nInteraction Matrix (Pearson Correlation Coefficient):")
    # Print header
    header_str = "    " + " ".join([f"{lbl:>5}" for lbl in short_labels])
    print(header_str)
    
    for i, name in enumerate(PART_NAMES):
        row_str = f"{short_labels[i]:<3}"
        for j in range(P):
            row_str += f" {corr_matrix[i, j]:5.2f}"
        print(row_str)
        
    # 4. Save matrix
    output_dir = os.path.join(REPO_ROOT, 'analysis', 'cross_body_interaction', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'interaction_matrix.npy')
    np.save(output_path, corr_matrix)
    print(f"\nSaved interaction matrix to: {output_path}")
    
    # 5. Extract distinct pairs (i < j)
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            pairs.append(((PART_NAMES[i], PART_NAMES[j]), corr_matrix[i, j]))
            
    # Sort pairs by correlation value
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Report Top 5 strongest
    print("\nTop 5 Strongest Body-Part Interactions:")
    for rank, ((p1, p2), corr) in enumerate(sorted_pairs[:5], 1):
        print(f"  {rank}. {p1} <-> {p2}: {corr:.4f}")
        
    # Report Top 5 weakest (from bottom up)
    print("\nTop 5 Weakest Body-Part Interactions:")
    for rank, ((p1, p2), corr) in enumerate(reversed(sorted_pairs[-5:]), 1):
        print(f"  {rank}. {p1} <-> {p2}: {corr:.4f}")

if __name__ == '__main__':
    main()
