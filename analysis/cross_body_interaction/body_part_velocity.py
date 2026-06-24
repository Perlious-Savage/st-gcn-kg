import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Setup import paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from net.body_part import BodyPartAggregator, PART_NAMES
from analysis.temporal_segmentation.motion_energy import resolve_data_path

def main():
    # 1. Load data
    data_path = resolve_data_path()
    data = np.load(data_path, mmap_mode='r')
    sample_idx = 0
    sample = np.array(data[sample_idx])  # (C, T, V, M)
    
    # Person-1 only
    skeleton_ctv = sample[:, :, :, 0]  # (3, T, V)
    skeleton_tvc = skeleton_ctv.transpose(1, 2, 0)  # (T, V, C)
    print("1. Skeleton shape (T, V, C):", skeleton_tvc.shape)
    
    # 2. Body part aggregation
    # BodyPartAggregator expects (N, C, T, V)
    # We can prepare the input as (1, 3, T, V)
    x_in = torch.tensor(skeleton_ctv).unsqueeze(0).float()
    print("   Aggregator input shape (1, C, T, V):", tuple(x_in.shape))
    
    aggregator = BodyPartAggregator()
    with torch.no_grad():
        y_out = aggregator(x_in)  # (1, C, T, P)
        
    print("   Aggregator output shape (1, C, T, P):", tuple(y_out.shape))
    
    # Convert back to (T, P, C) for analysis
    part_features = y_out.squeeze(0).permute(1, 2, 0).numpy()  # (T, P, C)
    print("2. Body-part features shape (T, P, C):", part_features.shape)
    
    # 3. Compute velocities
    # velocity_part[t] = part[t] - part[t-1], padded at t=0
    T, P, C = part_features.shape
    velocity_part = np.zeros_like(part_features)
    velocity_part[1:] = part_features[1:] - part_features[:-1]
    print("3. Velocity shape (T, P, C):", velocity_part.shape)
    
    # 4. Compute motion magnitudes (L2 norm across xyz)
    speed_part = np.linalg.norm(velocity_part, axis=-1)  # (T, P)
    print("4. Motion magnitude shape (T, P):", speed_part.shape)
    
    # 5. Save visualization
    output_dir = os.path.join(REPO_ROOT, 'analysis', 'cross_body_interaction', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'body_part_motion_sample0.png')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    frames = np.arange(T)
    for p_idx, name in enumerate(PART_NAMES):
        ax.plot(frames, speed_part[:, p_idx], label=name, alpha=0.8, linewidth=1.5)
        
    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Magnitude (L2 velocity)')
    ax.set_title('Body-Part Motion Magnitudes Over Time (Sample 0, Person 1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("5. Saved plot to:", output_path)
    
    # 6. Rank activity
    mean_magnitudes = speed_part.mean(axis=0)
    ranking = sorted(zip(PART_NAMES, mean_magnitudes), key=lambda x: x[1], reverse=True)
    
    print("\nMean motion magnitude per body part:")
    for name, score in ranking:
        print(f"  {name:<12}: {score:.6f}")
        
    print("\nBody-part activity ranking (most active -> least active):")
    print("  " + " -> ".join([name for name, _ in ranking]))

if __name__ == '__main__':
    main()
