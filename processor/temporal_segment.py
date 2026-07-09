#!/usr/bin/env python
"""
Temporal Micro-Action Segmentation for NTU skeleton sequences.

Deliverables:
  1. Onset frame   — first frame of each active phase
  2. Apex frame    — peak motion-energy frame within each active phase
  3. Offset frame  — last frame of each active phase
  4. Motion-energy plot with onset / apex / offset annotated
  5. Segmented skeleton clips saved as .npy files
  6. CSV summary: filename, onset, apex, offset

Pipeline:
  Load → velocity → energy → smooth → detect phases → export all deliverables

Usage (from repo root):
    python -m processor.temporal_segment --data <path_to_npy> --sample 0 --out_dir ./segments
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Step 2: Joint velocity
# ---------------------------------------------------------------------------

def compute_velocity(skeleton):
    """Finite-difference velocity along the time axis.

    Args:
        skeleton: (T, V, C) numpy array — one person's joint positions.

    Returns:
        velocity: (T-1, V, C) array of per-joint velocity vectors.
    """
    return np.diff(skeleton, axis=0)  # (T-1, V, C)


# ---------------------------------------------------------------------------
# Step 3: Motion energy
# ---------------------------------------------------------------------------

def compute_motion_energy(velocity):
    """Per-frame scalar motion energy = sum of joint speed magnitudes.

    Args:
        velocity: (T-1, V, C) array.

    Returns:
        energy: (T-1,) array — total kinetic proxy per frame.
    """
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, V)
    return speed.sum(axis=-1)                   # (T-1,)


# ---------------------------------------------------------------------------
# Step 4: Smooth motion energy (1-D Gaussian)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size, sigma):
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth_energy(energy, window=15, sigma=3.0):
    """Smooth energy curve with a 1-D Gaussian filter (no scipy dependency).

    Args:
        energy: (T,) array.
        window: kernel width (odd recommended).
        sigma:  Gaussian std in frames.

    Returns:
        smoothed: (T,) array.
    """
    kernel = _gaussian_kernel(window, sigma)
    # reflect-pad to reduce boundary artefacts
    pad = window // 2
    padded = np.pad(energy, pad, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')


# ---------------------------------------------------------------------------
# Step 5 & 6: Detect phases → temporal segments with onset / apex / offset
# ---------------------------------------------------------------------------

def detect_phases(energy_smooth, threshold_ratio=0.3, min_phase_len=5):
    """Label each frame as 'active' or 'rest' using a threshold on smoothed energy.

    Args:
        energy_smooth: (T,) smoothed energy.
        threshold_ratio: fraction of (max - min) above min to split phases.
        min_phase_len: merge phases shorter than this into neighbours.

    Returns:
        labels: (T,) array of ints — 0 = rest, 1 = active.
        segments: list of (start, end, label_str) tuples (end is inclusive).
    """
    lo, hi = energy_smooth.min(), energy_smooth.max()
    threshold = lo + threshold_ratio * (hi - lo)
    raw_labels = (energy_smooth >= threshold).astype(int)

    # --- build initial segments ---
    segments = []
    start = 0
    for i in range(1, len(raw_labels)):
        if raw_labels[i] != raw_labels[start]:
            segments.append((start, i - 1, raw_labels[start]))
            start = i
    segments.append((start, len(raw_labels) - 1, raw_labels[start]))

    # --- merge short segments into neighbours ---
    merged = True
    while merged:
        merged = False
        new_segments = []
        for seg in segments:
            s, e, lab = seg
            if (e - s + 1) < min_phase_len and len(new_segments) > 0:
                # absorb into previous segment
                prev_s, prev_e, prev_lab = new_segments[-1]
                new_segments[-1] = (prev_s, e, prev_lab)
                merged = True
            else:
                new_segments.append(seg)
        segments = new_segments

    # rebuild per-frame labels from merged segments
    labels = np.zeros(len(energy_smooth), dtype=int)
    for s, e, lab in segments:
        labels[s:e + 1] = lab

    label_map = {0: 'rest', 1: 'active'}
    named_segments = [(s, e, label_map[lab]) for s, e, lab in segments]
    return labels, named_segments


def find_keyframes(segments, energy_smooth):
    """For each active segment, find onset, apex, and offset frames.

    Args:
        segments: list of (start, end, label_str) from detect_phases.
        energy_smooth: (T,) smoothed energy array.

    Returns:
        actions: list of dicts with keys
            {onset, apex, offset, peak_energy, segment_idx}
            for each active phase.
    """
    actions = []
    for idx, (s, e, label) in enumerate(segments):
        if label != 'active':
            continue
        segment_energy = energy_smooth[s:e + 1]
        apex_local = int(np.argmax(segment_energy))
        actions.append({
            'onset': s,
            'apex': s + apex_local,
            'offset': e,
            'peak_energy': float(segment_energy[apex_local]),
            'segment_idx': idx,
        })
    return actions


# ---------------------------------------------------------------------------
# Step 1: Load skeleton sequence
# ---------------------------------------------------------------------------

def load_skeleton(path, sample_idx=0, person_idx=0):
    """Load one skeleton sequence from an NTU-format .npy file.

    Expected .npy shape: either
      (N, C, T, V, M) — full dataset batch, or
      (C, T, V, M)    — single sample.

    Returns:
        skeleton: (T, V, C) array for the requested sample & person.
        raw_sample: (C, T, V, M) the full untrimmed sample (for clip export).
        valid_len: int, number of non-zero frames.
    """
    data = np.load(path, mmap_mode='r')
    if data.ndim == 5:
        sample = np.array(data[sample_idx])  # (C, T, V, M)
    elif data.ndim == 4:
        sample = np.array(data)              # (C, T, V, M)
    else:
        raise ValueError(
            'Unexpected .npy shape {}; need 4-D or 5-D NTU layout'.format(
                data.shape))

    skeleton = sample[:, :, :, person_idx]  # (C, T, V)
    skeleton = skeleton.transpose(1, 2, 0)  # (T, V, C)

    # Trim trailing zero-padded frames
    frame_energy = np.abs(skeleton).sum(axis=(1, 2))
    nonzero = np.nonzero(frame_energy)[0]
    if len(nonzero) == 0:
        raise ValueError('Sample {} person {} is all zeros'.format(
            sample_idx, person_idx))
    valid_len = nonzero[-1] + 1
    skeleton = skeleton[:valid_len]
    return skeleton, sample, valid_len


# ---------------------------------------------------------------------------
# Deliverable 5: Save segmented skeleton clips
# ---------------------------------------------------------------------------

def save_clips(sample, valid_len, actions, out_dir, sample_idx):
    """Save each active segment as a separate .npy clip in NTU layout (C, T_clip, V, M).

    Args:
        sample: (C, T, V, M) full sample array.
        valid_len: actual non-padded length.
        actions: list of action dicts from find_keyframes.
        out_dir: output directory for clips.
        sample_idx: sample index for naming.

    Returns:
        clip_paths: list of saved file paths.
    """
    clips_dir = os.path.join(out_dir, 'clips')
    os.makedirs(clips_dir, exist_ok=True)

    clip_paths = []
    for i, act in enumerate(actions):
        # +1 because offset is inclusive; clamp to valid_len on the raw sample
        onset = act['onset']
        offset = min(act['offset'] + 1, valid_len, sample.shape[1])
        clip = sample[:, onset:offset, :, :]  # (C, T_clip, V, M)
        fname = 'sample{:04d}_action{:02d}_f{}-{}.npy'.format(
            sample_idx, i, act['onset'], act['offset'])
        path = os.path.join(clips_dir, fname)
        np.save(path, clip)
        clip_paths.append(path)
    return clip_paths


# ---------------------------------------------------------------------------
# Deliverable 6: CSV summary
# ---------------------------------------------------------------------------

def write_csv(actions, clip_paths, out_dir, sample_idx):
    """Write a CSV with filename, onset, apex, offset for each active phase.

    Args:
        actions: list of action dicts.
        clip_paths: list of saved .npy paths (parallel to actions).
        out_dir: output directory.
        sample_idx: sample index for the CSV filename.

    Returns:
        csv_path: path to the written CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'segments_sample{:04d}.csv'.format(sample_idx))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'onset', 'apex', 'offset', 'duration', 'peak_energy'])
        for act, clip_path in zip(actions, clip_paths):
            writer.writerow([
                os.path.basename(clip_path),
                act['onset'],
                act['apex'],
                act['offset'],
                act['offset'] - act['onset'] + 1,
                '{:.4f}'.format(act['peak_energy']),
            ])
    return csv_path


# ---------------------------------------------------------------------------
# Deliverable 4: Motion-energy plot with onset / apex / offset
# ---------------------------------------------------------------------------

def visualize(energy_raw, energy_smooth, segments, actions, save_path=None):
    """Plot motion energy with phase shading and onset/apex/offset markers."""
    T = len(energy_smooth)
    frames = np.arange(T)

    fig, ax = plt.subplots(figsize=(14, 5))

    # shade phases
    phase_colors = {'rest': '#2d3436', 'active': '#e17055'}
    for s, e, label in segments:
        ax.axvspan(s, e, alpha=0.12, color=phase_colors.get(label, '#636e72'))

    # energy curves
    ax.plot(frames, energy_raw[:T], color='#b2bec3', linewidth=0.8,
            label='raw energy', zorder=1)
    ax.plot(frames, energy_smooth, color='#d63031', linewidth=1.8,
            label='smoothed', zorder=2)

    # onset / apex / offset markers
    for i, act in enumerate(actions):
        tag = 'action {}'.format(i)
        ax.axvline(act['onset'], color='#00b894', linewidth=1.5,
                   linestyle='--', alpha=0.9, zorder=3)
        ax.axvline(act['offset'], color='#6c5ce7', linewidth=1.5,
                   linestyle='--', alpha=0.9, zorder=3)

        # apex dot
        ax.plot(act['apex'], energy_smooth[act['apex']],
                marker='v', markersize=10, color='#fdcb6e',
                markeredgecolor='#2d3436', markeredgewidth=1.2, zorder=4)

        # labels at top
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else energy_smooth.max() * 1.1
        ax.text(act['onset'], y_top * 0.95, 'onset', fontsize=7,
                color='#00b894', ha='center', va='top', fontweight='bold')
        ax.text(act['apex'], y_top * 0.88, 'apex', fontsize=7,
                color='#e17055', ha='center', va='top', fontweight='bold')
        ax.text(act['offset'], y_top * 0.95, 'offset', fontsize=7,
                color='#6c5ce7', ha='center', va='top', fontweight='bold')

    # legend entries for markers
    ax.plot([], [], color='#00b894', linestyle='--', linewidth=1.5, label='onset')
    ax.plot([], [], color='#6c5ce7', linestyle='--', linewidth=1.5, label='offset')
    ax.plot([], [], marker='v', color='#fdcb6e', linestyle='None',
            markersize=8, markeredgecolor='#2d3436', label='apex')

    ax.set_xlabel('Frame')
    ax.set_ylabel('Motion Energy')
    ax.set_title('Temporal Micro-Action Segmentation')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print('Saved plot to', save_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Temporal Micro-Action Segmentation')
    parser.add_argument('--data', required=True,
                        help='Path to NTU .npy skeleton file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index (for batched .npy)')
    parser.add_argument('--person', type=int, default=0,
                        help='Person index (0 or 1)')
    parser.add_argument('--window', type=int, default=15,
                        help='Smoothing window size')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Gaussian sigma for smoothing')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Energy threshold ratio for phase detection')
    parser.add_argument('--min_phase', type=int, default=5,
                        help='Minimum phase length in frames')
    parser.add_argument('--out_dir', default='./segments',
                        help='Output directory for clips, CSV, and plot')
    args = parser.parse_args()

    # Step 1: Load
    skeleton, raw_sample, valid_len = load_skeleton(args.data, args.sample, args.person)
    print('Loaded skeleton: {} frames, {} joints, {} coords'.format(*skeleton.shape))

    # Step 2: Velocity
    velocity = compute_velocity(skeleton)

    # Step 3: Motion energy
    energy_raw = compute_motion_energy(velocity)

    # Step 4: Smooth
    energy_smooth = smooth_energy(energy_raw, window=args.window, sigma=args.sigma)

    # Step 5 & 6: Detect phases → segments
    labels, segments = detect_phases(
        energy_smooth,
        threshold_ratio=args.threshold,
        min_phase_len=args.min_phase)

    # Find onset / apex / offset for active phases
    actions = find_keyframes(segments, energy_smooth)

    print('\nDetected {} segments ({} active):'.format(
        len(segments), len(actions)))
    for s, e, lab in segments:
        print('  frames {:>4d}-{:<4d}  ({:>3d} frames)  {}'.format(
            s, e, e - s + 1, lab))

    print('\nKeyframes:')
    for i, act in enumerate(actions):
        print('  action {}: onset={}, apex={}, offset={}, peak_energy={:.4f}'.format(
            i, act['onset'], act['apex'], act['offset'], act['peak_energy']))

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Deliverable 4: Motion-energy plot
    plot_path = os.path.join(args.out_dir, 'energy_sample{:04d}.png'.format(args.sample))
    visualize(energy_raw, energy_smooth, segments, actions, save_path=plot_path)

    # Deliverable 5: Segmented skeleton clips
    clip_paths = save_clips(raw_sample, valid_len, actions, args.out_dir, args.sample)
    for p in clip_paths:
        print('Saved clip:', p)

    # Deliverable 6: CSV summary
    csv_path = write_csv(actions, clip_paths, args.out_dir, args.sample)
    print('Saved CSV:', csv_path)

    print('\nAll deliverables written to', os.path.abspath(args.out_dir))


if __name__ == '__main__':
    main()
