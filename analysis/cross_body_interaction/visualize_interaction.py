import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

# Setup import paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from net.body_part import BodyPartAggregator, PART_NAMES
# Removed invalid list_dir import
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
    
    # 1. Get active phase
    res = process_sample(skeleton)
    valid_start = res['valid_start']
    onset, apex, offset = detect_phases_new(res['smoothed_energy'])
    orig_onset = int(onset + valid_start)
    orig_offset = int(offset + valid_start)
    
    # 2. Extract active part features
    active_skeleton = skeleton[orig_onset:orig_offset + 1]
    skeleton_ctv = active_skeleton.transpose(2, 0, 1)
    x_in = torch.tensor(skeleton_ctv).unsqueeze(0).float()
    
    aggregator = BodyPartAggregator()
    with torch.no_grad():
        y_out = aggregator(x_in)
        
    part_features = y_out.squeeze(0).permute(1, 2, 0).numpy()
    T_act, P, C = part_features.shape
    velocity_part = np.zeros_like(part_features)
    velocity_part[1:] = part_features[1:] - part_features[:-1]
    speed_part = np.linalg.norm(velocity_part, axis=-1)
    
    # 3. Active Pearson Correlation Matrix
    corr_matrix = np.corrcoef(speed_part, rowvar=False)
    
    # Save the matrix
    output_dir = os.path.join(REPO_ROOT, 'analysis', 'cross_body_interaction', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'active_interaction_matrix.npy'), corr_matrix)
    
    # PART A: HEATMAP
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    
    # Show labels
    ax.set_xticks(np.arange(len(PART_NAMES)))
    ax.set_yticks(np.arange(len(PART_NAMES)))
    ax.set_xticklabels(PART_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(PART_NAMES)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(PART_NAMES)):
        for j in range(len(PART_NAMES)):
            # Highlight strongest (excluding self-correlation of 1.0)
            text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                           ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white")
            
    # Highlight strongest interaction (left_arm <-> right_arm: 0.89)
    # The indices for left_arm and right_arm are 2 and 3 respectively
    rect = plt.Rectangle((1.5, 2.5), 2.0, 1.0, fill=False, edgecolor='gold', lw=3)
    # Wait, indices are: H=0, T=1, LA=2, RA=3, LL=4, RL=5
    # Strongest is LA (2) and RA (3).
    # Let's draw highlights on (2, 3) and (3, 2)
    ax.add_patch(plt.Rectangle((2.5, 1.5), 1.0, 1.0, fill=False, edgecolor='yellow', lw=3))
    ax.add_patch(plt.Rectangle((1.5, 2.5), 1.0, 1.0, fill=False, edgecolor='yellow', lw=3))
    
    plt.colorbar(im)
    ax.set_title("Active-Phase Pearson Correlation Heatmap (Sample 0)")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'interaction_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print("Heatmap saved to:", heatmap_path)
    
    # PART B: INTERACTION GRAPH
    G = nx.Graph()
    for name in PART_NAMES:
        G.add_node(name)
        
    for i in range(len(PART_NAMES)):
        for j in range(i + 1, len(PART_NAMES)):
            weight = corr_matrix[i, j]
            if weight > 0.5:
                G.add_edge(PART_NAMES[i], PART_NAMES[j], weight=weight)
                
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue', ax=ax)
    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    weights = [e[2]['weight'] * 4 for e in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', ax=ax)
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
    
    ax.set_title("Interaction Graph (Threshold > 0.5)")
    plt.axis('off')
    plt.tight_layout()
    graph_path = os.path.join(output_dir, 'interaction_graph.png')
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print("Graph saved to:", graph_path)
    
    # Output Graph stats
    print("\n--- Graph Statistics ---")
    print(f"Node count: {G.number_of_nodes()}")
    print(f"Edge count: {G.number_of_edges()}")
    print("Node Degrees:")
    for node, deg in G.degree():
        print(f"  {node:<12}: {deg}")
        
    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    if edges_sorted:
        print(f"Strongest edge: {edges_sorted[0][0]} <-> {edges_sorted[0][1]} ({edges_sorted[0][2]['weight']:.4f})")
        print(f"Weakest retained edge: {edges_sorted[-1][0]} <-> {edges_sorted[-1][1]} ({edges_sorted[-1][2]['weight']:.4f})")
        
    # PART C: COMPARE WITH CURRENT KG
    kg_edges = [
        ('head', 'torso'),
        ('torso', 'left_arm'),
        ('torso', 'right_arm'),
        ('torso', 'left_leg'),
        ('torso', 'right_leg'),
    ]
    
    print("\n--- KG Edge Correlation Comparison ---")
    print(f"{'KG Edge':<25} | {'Correlation':<12}")
    print("-" * 42)
    for u, v in kg_edges:
        u_idx = PART_NAMES.index(u)
        v_idx = PART_NAMES.index(v)
        corr = corr_matrix[u_idx, v_idx]
        print(f"{u} <-> {v:<15} | {corr:.4f}")

if __name__ == '__main__':
    main()
