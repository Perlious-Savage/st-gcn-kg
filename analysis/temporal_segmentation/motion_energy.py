#!/usr/bin/env python
"""Motion energy from a single NTU RGB+D validation skeleton (Task 01)."""

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
SAMPLE_INDEX = 0


def resolve_data_path():
    for path in (DEFAULT_DATA_PATH, FALLBACK_DATA_PATH):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        'Could not find val_data.npy. Tried:\n  {}\n  {}'.format(
            DEFAULT_DATA_PATH, FALLBACK_DATA_PATH))


def load_sample_person1(data_path, index=SAMPLE_INDEX):
    """Load one sample and return Person-1 skeleton as (T, V, C)."""
    data = np.load(data_path, mmap_mode='r')
    sample = np.array(data[index])  # (C, T, V, M)
    skeleton_ctv = sample[:, :, :, 0]  # Person-1: (C, T, V)
    skeleton = skeleton_ctv.transpose(1, 2, 0)  # (T, V, C)
    return skeleton


def compute_velocity(skeleton):
    """Joint velocity: velocity[t] = skeleton[t] - skeleton[t-1]."""
    velocity = np.zeros_like(skeleton)
    velocity[1:] = skeleton[1:] - skeleton[:-1]
    return velocity


def compute_motion_energy(velocity):
    """Per-frame motion energy = sum of joint L2 speeds over xyz."""
    speed = np.linalg.norm(velocity, axis=-1)  # (T, V)
    motion_energy = speed.sum(axis=-1)  # (T,)
    return motion_energy


def smooth_motion_energy(motion_energy, sigma=GAUSSIAN_SIGMA):
    return gaussian_filter1d(motion_energy, sigma=sigma)


def plot_motion_energy(motion_energy, smoothed, output_path):
    frames = np.arange(motion_energy.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, motion_energy, label='raw energy', alpha=0.7, linewidth=1.2)
    ax.plot(frames, smoothed, label='smoothed energy', linewidth=1.5)
    ax.set_xlabel('frame')
    ax.set_ylabel('motion energy')
    ax.set_title('NTU validation sample {} — Person 1 motion energy'.format(SAMPLE_INDEX))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    data_path = resolve_data_path()
    print('[data]', data_path)

    skeleton = load_sample_person1(data_path, index=SAMPLE_INDEX)
    velocity = compute_velocity(skeleton)
    motion_energy = compute_motion_energy(velocity)
    smoothed = smooth_motion_energy(motion_energy)

    print('skeleton shape:      ', skeleton.shape)
    print('velocity shape:      ', velocity.shape)
    print('motion energy shape: ', motion_energy.shape)

    output_path = os.path.join(OUTPUT_DIR, 'motion_energy_sample.png')
    plot_motion_energy(motion_energy, smoothed, output_path)
    print('[saved]', output_path)


if __name__ == '__main__':
    main()
