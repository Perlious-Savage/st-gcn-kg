#!/usr/bin/env python
"""Motion energy and temporal phase detection for NTU RGB+D skeletons."""

from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, 'ntu_processed', 'xsub', 'val_data.npy')
FALLBACK_DATA_PATH = os.path.join(
    os.path.expanduser('~'), 'Downloads', 'ntu_processed', 'xsub', 'val_data.npy')
GAUSSIAN_SIGMA = 3.0
PEAK_THRESHOLD_RATIO = 0.2
TEST_SAMPLE_INDICES = [0, 1, 2, 3, 4]


def resolve_data_path():
    for path in (DEFAULT_DATA_PATH, FALLBACK_DATA_PATH):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        'Could not find val_data.npy. Tried:\n  {}\n  {}'.format(
            DEFAULT_DATA_PATH, FALLBACK_DATA_PATH))


def load_sample_person1(data_path, index):
    """Load one sample and return Person-1 skeleton as (T, V, C)."""
    data = np.load(data_path, mmap_mode='r')
    sample = np.array(data[index])  # (C, T, V, M)
    skeleton_ctv = sample[:, :, :, 0]  # Person-1: (C, T, V)
    skeleton = skeleton_ctv.transpose(1, 2, 0)  # (T, V, C)
    return skeleton


def frame_validity_mask(skeleton):
    """True for frames where at least one joint has non-zero coordinates."""
    per_joint = np.sum(np.abs(skeleton), axis=-1)  # (T, V)
    return np.sum(per_joint, axis=1) > 0  # (T,)


def get_valid_range(valid_mask):
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return None, None
    return int(valid_indices[0]), int(valid_indices[-1])


def trim_skeleton(skeleton, valid_start, valid_end):
    return skeleton[valid_start:valid_end + 1]


def compute_velocity(skeleton):
    """Legacy velocity with zero-padded first frame (used by spike_diagnosis)."""
    velocity = np.zeros_like(skeleton)
    velocity[1:] = skeleton[1:] - skeleton[:-1]
    return velocity


def compute_velocity_trimmed(skeleton_trimmed):
    """Velocity from consecutive trimmed frames: diff along time axis."""
    if skeleton_trimmed.shape[0] < 2:
        return np.zeros((0, skeleton_trimmed.shape[1], skeleton_trimmed.shape[2]))
    return np.diff(skeleton_trimmed, axis=0)


def compute_motion_energy(velocity):
    """Per-step motion energy = sum of joint L2 speeds over xyz."""
    if velocity.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, V)
    return speed.sum(axis=-1)  # (T-1,)


def smooth_motion_energy(motion_energy, sigma=GAUSSIAN_SIGMA):
    if motion_energy.size == 0:
        return motion_energy.copy()
    return gaussian_filter1d(motion_energy, sigma=sigma)


def detect_phases(smoothed_energy, threshold_ratio=PEAK_THRESHOLD_RATIO):
    """Return onset, apex, offset indices in trimmed motion-energy space."""
    n = smoothed_energy.size
    if n == 0:
        return 0, 0, 0

    apex = int(np.argmax(smoothed_energy))
    peak = float(smoothed_energy[apex])
    threshold = threshold_ratio * peak

    onset_candidates = [i for i in range(apex) if smoothed_energy[i] > threshold]
    onset = onset_candidates[0] if onset_candidates else apex

    offset = n - 1
    for i in range(apex + 1, n):
        if smoothed_energy[i] < threshold:
            offset = i
            break

    return onset, apex, offset


def to_original_frame(index, valid_start):
    return int(index + valid_start)


def process_sample(skeleton, sigma=GAUSSIAN_SIGMA, threshold_ratio=PEAK_THRESHOLD_RATIO):
    """Full trimmed pipeline for one skeleton sequence."""
    valid_mask = frame_validity_mask(skeleton)
    valid_start, valid_end = get_valid_range(valid_mask)

    if valid_start is None:
        empty = np.zeros((0,), dtype=np.float64)
        return {
            'valid_start': None,
            'valid_end': None,
            'skeleton_trimmed': skeleton[:0],
            'velocity': np.zeros((0, skeleton.shape[1], skeleton.shape[2])),
            'motion_energy': empty,
            'smoothed_energy': empty,
            'onset': 0,
            'apex': 0,
            'offset': 0,
            'original_onset': 0,
            'original_apex': 0,
            'original_offset': 0,
            'original_frames': np.array([], dtype=int),
        }

    skeleton_trimmed = trim_skeleton(skeleton, valid_start, valid_end)
    velocity = compute_velocity_trimmed(skeleton_trimmed)
    motion_energy = compute_motion_energy(velocity)
    smoothed_energy = smooth_motion_energy(motion_energy, sigma=sigma)
    onset, apex, offset = detect_phases(smoothed_energy, threshold_ratio=threshold_ratio)

    original_frames = valid_start + np.arange(motion_energy.shape[0])

    return {
        'valid_start': valid_start,
        'valid_end': valid_end,
        'skeleton_trimmed': skeleton_trimmed,
        'velocity': velocity,
        'motion_energy': motion_energy,
        'smoothed_energy': smoothed_energy,
        'onset': onset,
        'apex': apex,
        'offset': offset,
        'original_onset': to_original_frame(onset, valid_start),
        'original_apex': to_original_frame(apex, valid_start),
        'original_offset': to_original_frame(offset, valid_start),
        'original_frames': original_frames,
    }


def plot_sample_phases(sample_index, result, output_path):
    motion_energy = result['motion_energy']
    smoothed = result['smoothed_energy']
    frames = result['original_frames']

    fig, ax = plt.subplots(figsize=(10, 4))
    if frames.size > 0:
        ax.plot(frames, motion_energy, label='raw energy', alpha=0.7, linewidth=1.2)
        ax.plot(frames, smoothed, label='smoothed energy', linewidth=1.5)

    phase_lines = (
        ('onset', result['original_onset'], 'green'),
        ('apex', result['original_apex'], 'red'),
        ('offset', result['original_offset'], 'blue'),
    )
    for label, frame, color in phase_lines:
        if result['valid_start'] is not None:
            ax.axvline(frame, color=color, linestyle='--', linewidth=1.5, label=label)

    ax.set_xlabel('frame (original)')
    ax.set_ylabel('motion energy')
    ax.set_title(
        'Sample {} — valid [{}, {}]'.format(
            sample_index, result['valid_start'], result['valid_end']))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_results_table(rows):
    header = '{:>6} {:>12} {:>10} {:>6} {:>6} {:>6}'.format(
        'sample', 'valid_start', 'valid_end', 'onset', 'apex', 'offset')
    print(header)
    print('-' * len(header))
    for row in rows:
        print('{:>6} {:>12} {:>10} {:>6} {:>6} {:>6}'.format(
            row['sample'],
            row['valid_start'] if row['valid_start'] is not None else 'n/a',
            row['valid_end'] if row['valid_end'] is not None else 'n/a',
            row['original_onset'],
            row['original_apex'],
            row['original_offset'],
        ))


def main():
    data_path = resolve_data_path()
    print('[data]', data_path)
    print()

    rows = []
    anomalies = []

    for sample_index in TEST_SAMPLE_INDICES:
        skeleton = load_sample_person1(data_path, index=sample_index)
        result = process_sample(skeleton)

        output_path = os.path.join(
            OUTPUT_DIR, 'motion_energy_sample_{}.png'.format(sample_index))
        plot_sample_phases(sample_index, result, output_path)

        row = {
            'sample': sample_index,
            'valid_start': result['valid_start'],
            'valid_end': result['valid_end'],
            'original_onset': result['original_onset'],
            'original_apex': result['original_apex'],
            'original_offset': result['original_offset'],
            'onset': result['onset'],
            'apex': result['apex'],
            'offset': result['offset'],
        }
        rows.append(row)

        if result['onset'] >= result['apex'] or result['offset'] <= result['apex']:
            anomalies.append(sample_index)

        print('[sample {}] skeleton {} -> trimmed {} | energy {} | saved {}'.format(
            sample_index,
            skeleton.shape,
            result['skeleton_trimmed'].shape,
            result['motion_energy'].shape,
            output_path,
        ))

    print()
    print_results_table(rows)

    print()
    if anomalies:
        print('Anomalies (onset >= apex OR offset <= apex): samples {}'.format(
            anomalies))
    else:
        print('No anomalies: all samples have onset < apex < offset.')

    print()
    print('[done] processed {} samples'.format(len(TEST_SAMPLE_INDICES)))


if __name__ == '__main__':
    main()
