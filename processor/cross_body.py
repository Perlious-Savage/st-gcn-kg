#!/usr/bin/env python
"""
Task 02: Cross-Body Interaction Modeling for NTU skeleton sequences.

Pipeline:
  1. Define NTU body parts       — reuse net.body_part partition
  2. Aggregate joint positions   — mean-pool raw coords per body part
  3. Compute body-part velocities— finite diff on part centroids
  4. Compute interaction matrix  — pairwise velocity correlation (6×6)
  5. Visualize interaction       — heatmaps per sample / action phase
  6. Convert to graph edges      — threshold on interaction strength
  7. Build NetworkX graph        — weighted interaction graph

Usage (from repo root):
    python -m processor.cross_body --data <npy> --sample 0 --out_dir ./cross_body
    python -m processor.cross_body --data <npy> --sample 0 --n_samples 10 --out_dir ./cross_body
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Repo root for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Inlined from net.body_part to avoid torch dependency for analysis scripts
PART_NAMES = ('head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
NTU_BODY_PARTS = {
    'head': [2, 3],
    'torso': [0, 1, 20],
    'left_arm': [4, 5, 6, 7, 21, 22],
    'right_arm': [8, 9, 10, 11, 23, 24],
    'left_leg': [12, 13, 14, 15],
    'right_leg': [16, 17, 18, 19],
}
NUM_PARTS = len(PART_NAMES)
from processor.temporal_segment import (
    load_skeleton, compute_velocity, compute_motion_energy,
    smooth_energy, detect_phases, find_keyframes,
)

# ---- Interaction pairs of interest ----
INTERACTION_PAIRS = [
    ('left_arm', 'right_arm'),
    ('head', 'torso'),
    ('left_arm', 'left_leg'),
    ('right_arm', 'right_leg'),
    ('left_arm', 'right_leg'),
    ('right_arm', 'left_leg'),
]

UPPER_BODY = ['head', 'torso', 'left_arm', 'right_arm']
LOWER_BODY = ['left_leg', 'right_leg']


# ---------------------------------------------------------------------------
# Step 2: Aggregate joints into body-part centroids
# ---------------------------------------------------------------------------

def aggregate_body_parts(skeleton):
    """Compute per-frame centroid for each body part.

    Args:
        skeleton: (T, V, C) array — joint positions for one person.

    Returns:
        centroids: (T, P, C) array — body-part centroids, P=6.
    """
    T, V, C = skeleton.shape
    centroids = np.zeros((T, NUM_PARTS, C))
    for p_idx, name in enumerate(PART_NAMES):
        joint_ids = NTU_BODY_PARTS[name]
        centroids[:, p_idx, :] = skeleton[:, joint_ids, :].mean(axis=1)
    return centroids


# ---------------------------------------------------------------------------
# Step 3: Compute body-part velocities
# ---------------------------------------------------------------------------

def compute_part_velocities(centroids):
    """Finite-difference velocity of body-part centroids.

    Args:
        centroids: (T, P, C) array.

    Returns:
        velocities: (T-1, P, C) array.
        speeds: (T-1, P) array — L2 speed per part per frame.
    """
    velocities = np.diff(centroids, axis=0)  # (T-1, P, C)
    speeds = np.linalg.norm(velocities, axis=-1)  # (T-1, P)
    return velocities, speeds


# ---------------------------------------------------------------------------
# Step 4: Compute interaction matrix
# ---------------------------------------------------------------------------

def compute_interaction_matrix(velocities, method='correlation'):
    """Pairwise interaction between body parts over a temporal window.

    Args:
        velocities: (T, P, C) array of body-part velocities.
        method: 'correlation' — Pearson correlation of speed profiles.
                'cosine'      — mean cosine similarity of velocity vectors.

    Returns:
        matrix: (P, P) symmetric interaction matrix in [0, 1].
    """
    T, P, C = velocities.shape
    matrix = np.zeros((P, P))

    if method == 'correlation':
        speeds = np.linalg.norm(velocities, axis=-1)  # (T, P)
        for i in range(P):
            for j in range(P):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    r = _safe_corrcoef(speeds[:, i], speeds[:, j])
                    matrix[i, j] = r
    elif method == 'cosine':
        for i in range(P):
            for j in range(P):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = _mean_cosine_sim(
                        velocities[:, i, :], velocities[:, j, :])
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return matrix


def _safe_corrcoef(a, b):
    """Pearson correlation, returning 0.0 for constant signals."""
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mean_cosine_sim(a, b):
    """Mean frame-wise cosine similarity between (T, C) arrays."""
    dot = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    denom = norm_a * norm_b
    valid = denom > 1e-8
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(dot[valid] / denom[valid]))


def compute_upper_lower_interaction(velocities):
    """Scalar interaction score between upper and lower body.

    Returns:
        score: float in [-1, 1] (correlation of aggregate speeds).
    """
    speeds = np.linalg.norm(velocities, axis=-1)  # (T, P)
    upper_idx = [list(PART_NAMES).index(n) for n in UPPER_BODY]
    lower_idx = [list(PART_NAMES).index(n) for n in LOWER_BODY]
    upper_speed = speeds[:, upper_idx].mean(axis=1)
    lower_speed = speeds[:, lower_idx].mean(axis=1)
    return _safe_corrcoef(upper_speed, lower_speed)


# ---------------------------------------------------------------------------
# Step 5: Visualize cross-body interaction heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(matrix, title='Cross-Body Interaction', save_path=None):
    """Plot a P×P interaction heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')

    ax.set_xticks(range(NUM_PARTS))
    ax.set_yticks(range(NUM_PARTS))
    ax.set_xticklabels(PART_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(PART_NAMES, fontsize=9)

    # annotate cells
    for i in range(NUM_PARTS):
        for j in range(NUM_PARTS):
            val = matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, '{:.2f}'.format(val), ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Interaction')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Step 6 & 7: Convert to graph edges + build NetworkX graph
# ---------------------------------------------------------------------------

def interaction_to_edges(matrix, threshold=0.3):
    """Convert interaction matrix to weighted edge list.

    Args:
        matrix: (P, P) symmetric matrix.
        threshold: minimum interaction to create an edge.

    Returns:
        edges: list of (part_a, part_b, weight) for above-threshold pairs.
    """
    edges = []
    for i in range(NUM_PARTS):
        for j in range(i + 1, NUM_PARTS):
            w = matrix[i, j]
            if w >= threshold:
                edges.append((PART_NAMES[i], PART_NAMES[j], float(w)))
    return edges


def build_and_plot_graph(edges, title='Interaction Graph', save_path=None):
    """Build a NetworkX graph from edges and visualize it.

    Args:
        edges: list of (node_a, node_b, weight).
        title: plot title.
        save_path: if set, save instead of showing.
    """
    try:
        import networkx as nx
    except ImportError:
        print('[WARN] networkx not installed; skipping graph visualization.')
        print('       Install with: pip install networkx')
        return None

    G = nx.Graph()
    G.add_nodes_from(PART_NAMES)
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Body-like layout for intuitive visualization
    pos = {
        'head':      (0.5, 1.0),
        'torso':     (0.5, 0.65),
        'left_arm':  (0.15, 0.75),
        'right_arm': (0.85, 0.75),
        'left_leg':  (0.3, 0.25),
        'right_leg': (0.7, 0.25),
    }

    # Edge widths and colors scaled by weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        max_w = max(edge_weights)
        edge_widths = [1.5 + 4.0 * (w / max_w) for w in edge_weights]
        edge_colors = plt.cm.OrRd([0.3 + 0.6 * (w / max_w) for w in edge_weights])
    else:
        edge_widths = []
        edge_colors = []

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1800,
                           node_color='#dfe6e9', edgecolors='#2d3436',
                           linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold',
                            font_color='#2d3436')
    if edge_weights:
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color=edge_colors, alpha=0.85)
        # Edge labels
        edge_labels = {(u, v): '{:.2f}'.format(G[u][v]['weight'])
                       for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=8, font_color='#d63031')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return G


# ---------------------------------------------------------------------------
# Multi-sample & per-phase analysis
# ---------------------------------------------------------------------------

def analyze_sample(data_path, sample_idx, person_idx, out_dir,
                   window=15, sigma=3.0, threshold_ratio=0.3,
                   min_phase=5, edge_threshold=0.3):
    """Run full cross-body analysis on one sample.

    Returns:
        result dict with matrices, actions, edges, etc.
    """
    # Load + temporal segmentation (reuse Task 01)
    skeleton, raw_sample, valid_len = load_skeleton(data_path, sample_idx, person_idx)
    T, V, C = skeleton.shape

    velocity = compute_velocity(skeleton)
    energy_raw = compute_motion_energy(velocity)
    energy_smooth = smooth_energy(energy_raw, window=window, sigma=sigma)
    _, segments = detect_phases(energy_smooth, threshold_ratio, min_phase)
    actions = find_keyframes(segments, energy_smooth)

    # Step 2: Body-part centroids
    centroids = aggregate_body_parts(skeleton)

    # Step 3: Body-part velocities
    part_vel, part_speeds = compute_part_velocities(centroids)

    # Step 4: Full-sequence interaction matrix
    full_matrix = compute_interaction_matrix(part_vel, method='correlation')
    upper_lower = compute_upper_lower_interaction(part_vel)

    # Per-action-phase interaction matrices
    phase_matrices = []
    for act in actions:
        onset, offset = act['onset'], act['offset']
        # Velocity indices are shifted by -1 from skeleton frames
        v_start = max(0, onset - 1)
        v_end = min(offset, part_vel.shape[0])
        if v_end - v_start < 3:
            phase_matrices.append(np.eye(NUM_PARTS))
            continue
        phase_vel = part_vel[v_start:v_end]
        pm = compute_interaction_matrix(phase_vel, method='correlation')
        phase_matrices.append(pm)

    # Step 5: Heatmaps
    sample_dir = os.path.join(out_dir, 'sample{:04d}'.format(sample_idx))
    os.makedirs(sample_dir, exist_ok=True)

    heatmap_path = os.path.join(sample_dir, 'heatmap_full.png')
    plot_heatmap(full_matrix,
                 title='Cross-Body Interaction (sample {}, full seq)'.format(sample_idx),
                 save_path=heatmap_path)

    phase_heatmap_paths = []
    for i, (pm, act) in enumerate(zip(phase_matrices, actions)):
        p = os.path.join(sample_dir, 'heatmap_action{:02d}.png'.format(i))
        plot_heatmap(pm,
                     title='Interaction — action {} (frames {}-{})'.format(
                         i, act['onset'], act['offset']),
                     save_path=p)
        phase_heatmap_paths.append(p)

    # Step 6 & 7: Graph
    edges = interaction_to_edges(full_matrix, threshold=edge_threshold)
    graph_path = os.path.join(sample_dir, 'interaction_graph.png')
    G = build_and_plot_graph(edges,
                             title='Interaction Graph (sample {})'.format(sample_idx),
                             save_path=graph_path)

    return {
        'sample_idx': sample_idx,
        'num_frames': T,
        'num_actions': len(actions),
        'actions': actions,
        'full_matrix': full_matrix,
        'phase_matrices': phase_matrices,
        'upper_lower_corr': upper_lower,
        'edges': edges,
        'heatmap_path': heatmap_path,
        'phase_heatmap_paths': phase_heatmap_paths,
        'graph_path': graph_path,
        'sample_dir': sample_dir,
    }


# ---------------------------------------------------------------------------
# Deliverable: CSV + report
# ---------------------------------------------------------------------------

def write_interaction_csv(results_list, out_dir):
    """Write interaction summary CSV across all analyzed samples."""
    csv_path = os.path.join(out_dir, 'cross_body_interactions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample', 'phase', 'onset', 'apex', 'offset',
            'upper_lower_corr',
            'left_arm_right_arm', 'head_torso',
            'left_arm_left_leg', 'right_arm_right_leg',
            'strongest_pair', 'strongest_corr',
        ])
        for res in results_list:
            si = res['sample_idx']
            # Full sequence row
            fm = res['full_matrix']
            strongest = _strongest_pair(fm)
            writer.writerow([
                si, 'full', '', '', '',
                '{:.4f}'.format(res['upper_lower_corr']),
                '{:.4f}'.format(fm[_pi('left_arm'), _pi('right_arm')]),
                '{:.4f}'.format(fm[_pi('head'), _pi('torso')]),
                '{:.4f}'.format(fm[_pi('left_arm'), _pi('left_leg')]),
                '{:.4f}'.format(fm[_pi('right_arm'), _pi('right_leg')]),
                strongest[0], '{:.4f}'.format(strongest[1]),
            ])
            # Per-action rows
            for j, (pm, act) in enumerate(zip(res['phase_matrices'], res['actions'])):
                strongest = _strongest_pair(pm)
                writer.writerow([
                    si, 'action_{}'.format(j),
                    act['onset'], act['apex'], act['offset'],
                    '',
                    '{:.4f}'.format(pm[_pi('left_arm'), _pi('right_arm')]),
                    '{:.4f}'.format(pm[_pi('head'), _pi('torso')]),
                    '{:.4f}'.format(pm[_pi('left_arm'), _pi('left_leg')]),
                    '{:.4f}'.format(pm[_pi('right_arm'), _pi('right_leg')]),
                    strongest[0], '{:.4f}'.format(strongest[1]),
                ])
    return csv_path


def _pi(name):
    """Part index by name."""
    return list(PART_NAMES).index(name)


def _strongest_pair(matrix):
    """Find the strongest off-diagonal interaction pair."""
    best_val = -1.0
    best_pair = ''
    for i in range(NUM_PARTS):
        for j in range(i + 1, NUM_PARTS):
            if matrix[i, j] > best_val:
                best_val = matrix[i, j]
                best_pair = '{}<->{}'.format(PART_NAMES[i], PART_NAMES[j])
    return best_pair, best_val


def write_report(results_list, out_dir):
    """Generate a short text report summarizing cross-body interactions."""
    report_path = os.path.join(out_dir, 'cross_body_report.txt')
    lines = []
    lines.append('=' * 60)
    lines.append('Cross-Body Interaction Report')
    lines.append('=' * 60)

    # Aggregate interaction matrices
    all_matrices = np.stack([r['full_matrix'] for r in results_list])
    mean_matrix = all_matrices.mean(axis=0)

    lines.append('\n--- Mean Interaction Matrix (across {} samples) ---'.format(
        len(results_list)))
    header = '{:>12s}'.format('') + ''.join(
        '{:>12s}'.format(n[:8]) for n in PART_NAMES)
    lines.append(header)
    for i, name in enumerate(PART_NAMES):
        row = '{:>12s}'.format(name[:8])
        row += ''.join('{:>12.3f}'.format(mean_matrix[i, j])
                       for j in range(NUM_PARTS))
        lines.append(row)

    # Strongest pairs globally
    lines.append('\n--- Strongest Interactions (mean across samples) ---')
    pairs_ranked = []
    for i in range(NUM_PARTS):
        for j in range(i + 1, NUM_PARTS):
            pairs_ranked.append((PART_NAMES[i], PART_NAMES[j], mean_matrix[i, j]))
    pairs_ranked.sort(key=lambda x: x[2], reverse=True)
    for a, b, v in pairs_ranked:
        lines.append('  {:<15s} <-> {:<15s}  corr = {:.4f}'.format(a, b, v))

    # Upper vs lower body
    mean_ul = np.mean([r['upper_lower_corr'] for r in results_list])
    lines.append('\n--- Upper Body <-> Lower Body ---')
    lines.append('  Mean correlation: {:.4f}'.format(mean_ul))

    # Per-sample summaries
    lines.append('\n--- Per-Sample Summaries ---')
    for res in results_list:
        strongest = _strongest_pair(res['full_matrix'])
        lines.append('  Sample {:4d}: {:3d} frames, {} actions, '
                     'strongest = {} ({:.3f}), upper<->lower = {:.3f}'.format(
                         res['sample_idx'], res['num_frames'],
                         res['num_actions'], strongest[0], strongest[1],
                         res['upper_lower_corr']))

    # Integration note
    lines.append('\n--- Integration with KG-STGCN ---')
    lines.append('  The interaction matrix reveals which body-part pairs move')
    lines.append('  in coordination. This can enhance KG-STGCN in two ways:')
    lines.append('  1. Dynamic edge weighting: use the interaction correlation')
    lines.append('     as edge weights in kg_gnn.py instead of fixed binary')
    lines.append('     adjacency — stronger interactions get heavier message')
    lines.append('     passing, letting the model attend to action-relevant')
    lines.append('     body-part connections.')
    lines.append('  2. Temporal attention: per-phase interaction matrices')
    lines.append('     differ from the full-sequence matrix. Feeding phase-')
    lines.append('     specific interaction features as auxiliary input can')
    lines.append('     help the model distinguish sub-actions within a clip.')
    lines.append('=' * 60)

    report = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cross-Body Interaction Modeling')
    parser.add_argument('--data', required=True,
                        help='Path to NTU .npy skeleton file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Starting sample index')
    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of samples to analyze')
    parser.add_argument('--person', type=int, default=0,
                        help='Person index (0 or 1)')
    parser.add_argument('--window', type=int, default=15,
                        help='Smoothing window for energy')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian sigma for smoothing')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Phase detection threshold ratio')
    parser.add_argument('--edge_threshold', type=float, default=0.3,
                        help='Minimum correlation to create a graph edge')
    parser.add_argument('--min_phase', type=int, default=5,
                        help='Minimum phase length in frames')
    parser.add_argument('--out_dir', default='./cross_body',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    for i in range(args.n_samples):
        idx = args.sample + i
        print('\n--- Analyzing sample {} ---'.format(idx))
        try:
            res = analyze_sample(
                args.data, idx, args.person, args.out_dir,
                window=args.window, sigma=args.sigma,
                threshold_ratio=args.threshold,
                min_phase=args.min_phase,
                edge_threshold=args.edge_threshold)
            results.append(res)
            print('  {} frames, {} actions, {} graph edges'.format(
                res['num_frames'], res['num_actions'], len(res['edges'])))
        except Exception as e:
            print('  [SKIP] sample {}: {}'.format(idx, e))

    if not results:
        print('No samples analyzed.')
        return

    # Deliverables
    csv_path = write_interaction_csv(results, args.out_dir)
    print('\nCSV:', csv_path)

    report_path = write_report(results, args.out_dir)
    print('Report:', report_path)

    print('\nAll deliverables written to', os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()
