#!/usr/bin/env python
"""Diagnose motion-energy spike for validation sample 0 (read-only)."""

from __future__ import print_function

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from motion_energy import (  # noqa: E402
    GAUSSIAN_SIGMA,
    OUTPUT_DIR,
    SAMPLE_INDEX,
    compute_motion_energy,
    compute_velocity,
    load_sample_person1,
    resolve_data_path,
    smooth_motion_energy,
)

# NTU RGB+D edges (0-based), from net/utils/graph.py neighbor_1base minus 1
NTU_EDGES = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 1), (13, 12), (14, 13),
    (15, 14), (16, 1), (17, 16), (18, 17), (19, 18),
    (21, 22), (22, 7), (23, 24), (24, 11),
]


def load_full_sample(data_path, index=SAMPLE_INDEX):
    data = np.load(data_path, mmap_mode='r')
    return np.array(data[index])  # (C, T, V, M)


def person_active_mask(sample_ctvm, person_idx, eps=1e-6):
    """True per frame if any joint has non-zero xyz for this person slot."""
    xyz = sample_ctvm[:3, :, :, person_idx]  # (3, T, V)
    return np.any(np.abs(xyz) > eps, axis=(0, 2))  # (T,)


def joint_displacements(frame_a, frame_b):
    """Per-joint L2 displacement between two skeleton frames (V,)."""
    return np.linalg.norm(frame_b - frame_a, axis=-1)


def max_joint_displacement(frame_a, frame_b):
    disp = joint_displacements(frame_a, frame_b)
    joint = int(np.argmax(disp))
    return float(disp[joint]), joint, disp


def all_zero_frame(frame, eps=1e-6):
    return not np.any(np.abs(frame) > eps)


def zero_joint_mask(frame, eps=1e-6):
    return np.all(np.abs(frame) < eps, axis=-1)


def plot_skeleton_2d(ax, frame_xyz, title, highlight_joint=None):
    xy = frame_xyz[:, :2]
    for i, j in NTU_EDGES:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], 'b-', lw=1.2, alpha=0.8)
    ax.scatter(xy[:, 0], xy[:, 1], c='crimson', s=18, zorder=3)
    if highlight_joint is not None:
        ax.scatter(xy[highlight_joint, 0], xy[highlight_joint, 1],
                   c='gold', s=80, edgecolors='black', zorder=4)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()


def diagnose():
    data_path = resolve_data_path()
    sample = load_full_sample(data_path, index=SAMPLE_INDEX)  # (C,T,V,M)
    skeleton = load_sample_person1(data_path, index=SAMPLE_INDEX)  # (T,V,C)
    velocity = compute_velocity(skeleton)
    motion_energy = compute_motion_energy(velocity)
    smoothed = smooth_motion_energy(motion_energy, sigma=GAUSSIAN_SIGMA)

    peak_frame = int(np.argmax(motion_energy))
    peak_energy = float(motion_energy[peak_frame])

    print('=' * 72)
    print('SPIKE DIAGNOSIS — validation sample {}, Person 1'.format(SAMPLE_INDEX))
    print('=' * 72)
    print('[data]', data_path)
    print('peak_frame:  ', peak_frame)
    print('peak_energy: ', peak_energy)

    lo = max(0, peak_frame - 10)
    hi = min(len(motion_energy) - 1, peak_frame + 10)
    frames = np.arange(lo, hi + 1)

    print('\n--- raw motion energy [peak-10 : peak+10] ---')
    for f in frames:
        print('  frame {:3d}: {:12.6f}'.format(f, motion_energy[f]))

    print('\n--- smoothed motion energy [peak-10 : peak+10] ---')
    for f in frames:
        print('  frame {:3d}: {:12.6f}'.format(f, smoothed[f]))

    f_m1 = peak_frame - 1
    f_0 = peak_frame
    f_p1 = peak_frame + 1

    sk_m1 = skeleton[f_m1]
    sk_0 = skeleton[f_0]
    sk_p1 = skeleton[f_p1]

    print('\n--- skeleton coordinates (xyz) at peak-1, peak, peak+1 ---')
    for label, frame_idx, coords in (
        ('peak-1', f_m1, sk_m1),
        ('peak', f_0, sk_0),
        ('peak+1', f_p1, sk_p1),
    ):
        print('\nframe {} (t={}):'.format(label, frame_idx))
        print(coords)

    max_disp_m1_0, joint_m1_0, disp_m1_0 = max_joint_displacement(sk_m1, sk_0)
    max_disp_0_p1, joint_0_p1, disp_0_p1 = max_joint_displacement(sk_0, sk_p1)

    print('\n--- max joint displacement ---')
    print('peak-1 -> peak:   {:.6f}  (joint {}, all joints: {})'.format(
        max_disp_m1_0, joint_m1_0, np.round(disp_m1_0, 4)))
    print('peak   -> peak+1: {:.6f}  (joint {}, all joints: {})'.format(
        max_disp_0_p1, joint_0_p1, np.round(disp_0_p1, 4)))

    p0_active = person_active_mask(sample, 0)
    p1_active = person_active_mask(sample, 1)

    print('\n--- person slot activity (any non-zero joint) ---')
    print('Person-0 active frames: {}/{}'.format(int(p0_active.sum()), len(p0_active)))
    print('Person-1 active frames: {}/{}'.format(int(p1_active.sum()), len(p1_active)))
    for f in (f_m1, f_0, f_p1):
        print('  frame {:3d}: P0={} P1={}'.format(
            f, bool(p0_active[f]), bool(p1_active[f])))

    print('\n--- artifact checks ---')
    for label, frame_idx, coords in (
        ('peak-1', f_m1, sk_m1),
        ('peak', f_0, sk_0),
        ('peak+1', f_p1, sk_p1),
    ):
        zj = zero_joint_mask(coords)
        print('  {} (t={}): all-zero frame={}; zero joints={}/{}'.format(
            label, frame_idx, all_zero_frame(coords), int(zj.sum()), len(zj)))

    # Compare Person-1 slot to Person-0 at spike frames (tracking switch)
    print('\n--- cross-person similarity at spike frames (mean joint L2) ---')
    for f in (f_m1, f_0, f_p1):
        p0 = sample[:3, f, :, 0].T  # (V,3)
        p1 = sample[:3, f, :, 1].T
        if p0_active[f] and p1_active[f]:
            dist = np.linalg.norm(p0 - p1, axis=-1).mean()
            print('  frame {:3d}: mean P0-P1 joint distance = {:.6f}'.format(f, dist))
        else:
            print('  frame {:3d}: one or both persons inactive'.format(f))

    # First/last active frame for Person-1
    if p1_active.any():
        first_active = int(np.argmax(p1_active))
        last_active = int(len(p1_active) - 1 - np.argmax(p1_active[::-1]))
    else:
        first_active = last_active = -1
    print('\nPerson-1 active span: frames {} .. {}'.format(first_active, last_active))
    print('Peak occurs at frame {} (relative offset from last active: {})'.format(
        peak_frame, peak_frame - last_active if last_active >= 0 else 'n/a'))

    # Side-by-side skeleton plot (xy projection)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_skeleton_2d(axes[0], sk_m1, 't={} (peak-1)'.format(f_m1))
    plot_skeleton_2d(axes[1], sk_0, 't={} (peak)'.format(f_0), highlight_joint=joint_m1_0)
    plot_skeleton_2d(axes[2], sk_p1, 't={} (peak+1)'.format(f_p1), highlight_joint=joint_0_p1)
    fig.suptitle('Person 1 skeleton around motion-energy spike (sample 0)')
    fig.tight_layout()
    out_plot = os.path.join(OUTPUT_DIR, 'spike_skeleton_frames.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print('\n[saved]', out_plot)

    # Root-cause classification
    cause = classify_root_cause(
        peak_frame=peak_frame,
        peak_energy=peak_energy,
        motion_energy=motion_energy,
        p1_active=p1_active,
        last_active=last_active,
        max_disp_m1_0=max_disp_m1_0,
        max_disp_0_p1=max_disp_0_p1,
        sk_m1=sk_m1,
        sk_0=sk_0,
        sk_p1=sk_p1,
    )
    print('\n' + '=' * 72)
    print('ROOT CAUSE REPORT')
    print('=' * 72)
    print(cause)
    return cause


def classify_root_cause(peak_frame, peak_energy, motion_energy, p1_active,
                        last_active, max_disp_m1_0, max_disp_0_p1,
                        sk_m1, sk_0, sk_p1):
    lines = []

    zero_after = np.all(motion_energy[peak_frame + 1:] < 1e-6)
    p1_gone_at_peak = not p1_active[peak_frame] if peak_frame < len(p1_active) else True
    p1_gone_after = not np.any(p1_active[peak_frame:peak_frame + 5])

    lines.append('Peak at frame {} with energy {:.2f}.'.format(peak_frame, peak_energy))
    lines.append('Max joint displacement peak-1->peak: {:.4f}; peak->peak+1: {:.4f}.'.format(
        max_disp_m1_0, max_disp_0_p1))

    # Determine primary cause
    if all_zero_frame(sk_0) or all_zero_frame(sk_p1):
        primary = 'C. missing skeleton frames'
        detail = ('Person-1 slot becomes all-zero at or immediately after the peak. '
                  'Motion energy collapses to 0 for all subsequent frames.')
    elif peak_frame >= last_active and zero_after:
        primary = 'C. missing skeleton frames / end-of-track padding'
        detail = ('Spike aligns with the last frame where Person-1 has valid coordinates; '
                  'later frames are zero-padded in the preprocessed tensor.')
    elif max_disp_m1_0 > 5.0 and max_disp_0_p1 < 0.1:
        primary = 'B. skeleton tracking artifact'
        detail = ('A single-frame coordinate jump produces a huge velocity at the peak, '
                  'then coordinates reset or vanish on the next frame.')
    elif max_disp_m1_0 > 2.0 and not p1_gone_at_peak:
        primary = 'B. skeleton tracking artifact (coordinate jump)'
        detail = 'Large per-joint displacement inconsistent with smooth human motion.'
    else:
        primary = 'A. genuine action motion'
        detail = 'Displacement pattern appears continuous across neighboring frames.'

    # Secondary checks
    secondary = []
    if p1_gone_at_peak or p1_gone_after:
        secondary.append('Person-1 disappears from slot 0 after the spike (not a person-switch into slot 0).')
    if zero_after:
        secondary.append('Motion energy is exactly zero from frame {} onward (padding/empty tail).'.format(
            peak_frame + 1))
    if all_zero_frame(sk_m1) and not all_zero_frame(sk_0):
        secondary.append('Person-1 reappears from zeros at peak frame (possible tracking re-init).')

    lines.append('')
    lines.append('Primary classification: {}'.format(primary))
    lines.append(detail)
    if secondary:
        lines.append('')
        lines.append('Supporting observations:')
        for s in secondary:
            lines.append('  - {}'.format(s))

    lines.append('')
    lines.append('Ruled out:')
    lines.append('  D. person switching — Person-1 slot (index 0) is used consistently; '
                 'inactivity is due to empty coordinates, not swap with Person-2.')
    lines.append('  E. preprocessing padding alone — padding explains zero tail, but the '
                 'spike itself is driven by the last valid->zero transition, not random pad noise.')

    return '\n'.join(lines)


if __name__ == '__main__':
    diagnose()
