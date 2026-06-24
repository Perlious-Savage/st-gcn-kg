import os
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

# Setup import paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from net.body_part import BodyPartAggregator, PART_NAMES
from analysis.temporal_segmentation.motion_energy import (
    resolve_data_path,
    load_sample_person1,
    process_sample,
)

# New threshold calculation
def detect_phases_new(smoothed_energy, threshold_ratio=0.2):
    n = smoothed_energy.size
    if n == 0:
        return 0, 0, 0
    apex = int(np.argmax(smoothed_energy))
    peak = float(smoothed_energy[apex])
    baseline = float(np.percentile(smoothed_energy, 10))
    threshold = baseline + threshold_ratio * (peak - baseline)
    onset_candidates = [i for i in range(apex) if smoothed_energy[i] > threshold]
    onset = onset_candidates[0] if onset_candidates else apex
    offset = n - 1
    for i in range(apex + 1, n):
        if smoothed_energy[i] < threshold:
            offset = i
            break
    return onset, apex, offset

def main():
    data_path = resolve_data_path()
    skeleton = load_sample_person1(data_path, index=0)
    
    # 1. Get onset and offset from the new pipeline
    # process_sample returns valid_start, etc.
    res = process_sample(skeleton)
    valid_start = res['valid_start']
    
    # Compute new onset and offset indices
    onset, apex, offset = detect_phases_new(res['smoothed_energy'])
    orig_onset = int(onset + valid_start)
    orig_offset = int(offset + valid_start)
    
    print(f"Active frames (original index): {orig_onset} to {orig_offset}")
    
    # Slice the original skeleton to active frames
    # shape of skeleton: (T, V, C)
    active_skeleton = skeleton[orig_onset:orig_offset + 1]
    print("Active skeleton shape:", active_skeleton.shape)
    
    # 2. Body part aggregation on active skeleton
    # Convert active_skeleton to (1, C, T_active, V)
    # skeleton is (T_active, V, C) -> transpose to (C, T_active, V)
    skeleton_ctv = active_skeleton.transpose(2, 0, 1)
    x_in = torch.tensor(skeleton_ctv).unsqueeze(0).float()
    
    aggregator = BodyPartAggregator()
    with torch.no_grad():
        y_out = aggregator(x_in)  # (1, C, T_active, P)
        
    part_features = y_out.squeeze(0).permute(1, 2, 0).numpy()  # (T_active, P, C)
    print("Part features shape:", part_features.shape)
    
    # 3. Compute velocities and motion magnitudes
    T_act, P, C = part_features.shape
    velocity_part = np.zeros_like(part_features)
    velocity_part[1:] = part_features[1:] - part_features[:-1]
    speed_part = np.linalg.norm(velocity_part, axis=-1)  # (T_active, P)
    
    # 4. Pearson correlation matrix
    pearson_matrix = np.corrcoef(speed_part, rowvar=False)
    
    # 5. Spearman correlation matrix
    spearman_matrix, _ = spearmanr(speed_part, axis=0)
    
    # 6. Print matrices
    short_labels = ["H", "T", "LA", "RA", "LL", "RL"]
    
    print("\n--- Pearson Correlation Matrix (Active Phase) ---")
    print("    " + " ".join([f"{lbl:>5}" for lbl in short_labels]))
    for i in range(P):
        row_str = f"{short_labels[i]:<3}"
        for j in range(P):
            row_str += f" {pearson_matrix[i, j]:5.2f}"
        print(row_str)
        
    print("\n--- Spearman Correlation Matrix (Active Phase) ---")
    print("    " + " ".join([f"{lbl:>5}" for lbl in short_labels]))
    for i in range(P):
        row_str = f"{short_labels[i]:<3}"
        for j in range(P):
            row_str += f" {spearman_matrix[i, j]:5.2f}"
        print(row_str)
        
    # Get distinct pairs for reporting (Pearson)
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            pairs.append(((PART_NAMES[i], PART_NAMES[j]), pearson_matrix[i, j]))
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Report Top 5 strongest
    print("\nTop 5 Strongest Pearson Interactions (Active Phase):")
    for rank, ((p1, p2), corr) in enumerate(sorted_pairs[:5], 1):
        print(f"  {rank}. {p1} <-> {p2}: {corr:.4f}")
        
    # Report Top 5 weakest
    print("\nTop 5 Weakest Pearson Interactions (Active Phase):")
    for rank, ((p1, p2), corr) in enumerate(reversed(sorted_pairs[-5:]), 1):
        print(f"  {rank}. {p1} <-> {p2}: {corr:.4f}")

if __name__ == '__main__':
    main()
