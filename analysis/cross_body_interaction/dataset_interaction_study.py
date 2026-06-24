import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    data = np.load(data_path, mmap_mode='r')
    num_samples = len(data)
    print(f"Total validation samples: {num_samples}")
    
    # Select 20 random validation sample indices using a fixed seed for reproducibility
    np.random.seed(42)
    sample_indices = np.random.choice(num_samples, 20, replace=False)
    print("Selected sample indices:", list(sample_indices))
    
    aggregator = BodyPartAggregator()
    corr_matrices = []
    
    processed_count = 0
    for idx in sample_indices:
        idx = int(idx)
        skeleton = load_sample_person1(data_path, index=idx)
        
        # 1. Temporal Segmentation
        res = process_sample(skeleton)
        valid_start = res['valid_start']
        if valid_start is None:
            print(f"Sample {idx} is empty/invalid, skipping.")
            continue
            
        onset, apex, offset = detect_phases_new(res['smoothed_energy'])
        orig_onset = int(onset + valid_start)
        orig_offset = int(offset + valid_start)
        
        # Slice to active frames
        active_skeleton = skeleton[orig_onset:orig_offset + 1]
        if len(active_skeleton) < 3:
            # Not enough frames to compute correlation
            print(f"Sample {idx} active phase is too short ({len(active_skeleton)} frames), skipping.")
            continue
            
        # 2. Aggregate
        skeleton_ctv = active_skeleton.transpose(2, 0, 1)
        x_in = torch.tensor(skeleton_ctv).unsqueeze(0).float()
        with torch.no_grad():
            y_out = aggregator(x_in)
            
        part_features = y_out.squeeze(0).permute(1, 2, 0).numpy()
        
        # 3. Velocities and motion magnitudes
        T_act, P, C = part_features.shape
        velocity_part = np.zeros_like(part_features)
        velocity_part[1:] = part_features[1:] - part_features[:-1]
        speed_part = np.linalg.norm(velocity_part, axis=-1)
        
        # 4. Pearson Correlation Matrix
        corr_matrix = np.corrcoef(speed_part, rowvar=False)
        
        # Check for NaNs (e.g. if a body part has exactly zero movement throughout the segment)
        if np.isnan(corr_matrix).any():
            # Fill NaNs with 0
            corr_matrix = np.nan_to_num(corr_matrix)
            
        corr_matrices.append(corr_matrix)
        processed_count += 1
        
    print(f"Successfully processed {processed_count} samples.")
    
    # Convert list of matrices to a 3D numpy array: (processed_count, 6, 6)
    corr_stack = np.stack(corr_matrices, axis=0)
    
    # 5. Compute mean and std matrices
    mean_matrix = np.mean(corr_stack, axis=0)
    std_matrix = np.std(corr_stack, axis=0)
    
    # Save outputs
    output_dir = os.path.join(REPO_ROOT, 'analysis', 'cross_body_interaction', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Heatmaps
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(PART_NAMES)))
    ax.set_yticks(np.arange(len(PART_NAMES)))
    ax.set_xticklabels(PART_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(PART_NAMES)
    for i in range(len(PART_NAMES)):
        for j in range(len(PART_NAMES)):
            ax.text(j, i, f"{mean_matrix[i, j]:.2f}", ha="center", va="center", color="black" if abs(mean_matrix[i, j]) < 0.7 else "white")
    plt.colorbar(im)
    ax.set_title("Dataset Mean Active Pearson Correlation (20 Samples)")
    plt.tight_layout()
    mean_heatmap_path = os.path.join(output_dir, 'mean_interaction_heatmap.png')
    plt.savefig(mean_heatmap_path, dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(std_matrix, cmap='viridis', vmin=0.0)
    ax.set_xticks(np.arange(len(PART_NAMES)))
    ax.set_yticks(np.arange(len(PART_NAMES)))
    ax.set_xticklabels(PART_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(PART_NAMES)
    for i in range(len(PART_NAMES)):
        for j in range(len(PART_NAMES)):
            ax.text(j, i, f"{std_matrix[i, j]:.2f}", ha="center", va="center", color="black" if std_matrix[i, j] < 0.3 else "white")
    plt.colorbar(im)
    ax.set_title("Dataset Std Active Pearson Correlation (20 Samples)")
    plt.tight_layout()
    std_heatmap_path = os.path.join(output_dir, 'std_interaction_heatmap.png')
    plt.savefig(std_heatmap_path, dpi=150)
    plt.close()
    
    print("Saved mean heatmap to:", mean_heatmap_path)
    print("Saved std heatmap to:", std_heatmap_path)
    
    # 6. Extract unique pairs and rank
    pairs = []
    for i in range(len(PART_NAMES)):
        for j in range(i + 1, len(PART_NAMES)):
            pairs.append(((PART_NAMES[i], PART_NAMES[j]), mean_matrix[i, j], std_matrix[i, j]))
            
    # Rank by mean correlation
    sorted_by_mean = sorted(pairs, key=lambda x: x[1], reverse=True)
    
    # Rank by std (variance)
    sorted_by_std = sorted(pairs, key=lambda x: x[2], reverse=True)
    
    print("\n--- Summary Results ---")
    print(f"1. Average strongest interaction pair: {sorted_by_mean[0][0][0]} <-> {sorted_by_mean[0][0][1]} (mean={sorted_by_mean[0][1]:.4f}, std={sorted_by_mean[0][2]:.4f})")
    print(f"2. Average weakest interaction pair: {sorted_by_mean[-1][0][0]} <-> {sorted_by_mean[-1][0][1]} (mean={sorted_by_mean[-1][1]:.4f}, std={sorted_by_mean[-1][2]:.4f})")
    
    print("\n3. Top 10 interactions ranked by mean correlation:")
    for rank, ((p1, p2), mean_val, std_val) in enumerate(sorted_by_mean[:10], 1):
        print(f"  {rank:2d}. {p1:<12} <-> {p2:<12} | Mean: {mean_val:.4f} | Std: {std_val:.4f}")
        
    print("\n4. Interactions with highest variance (Std):")
    for rank, ((p1, p2), mean_val, std_val) in enumerate(sorted_by_std[:5], 1):
        print(f"  {rank:2d}. {p1:<12} <-> {p2:<12} | Std: {std_val:.4f} | Mean: {mean_val:.4f}")

if __name__ == '__main__':
    main()
