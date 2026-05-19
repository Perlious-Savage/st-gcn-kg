"""
Semantic body-part aggregation for ST-GCN joint-level features (NTU RGB+D, V=25).

Input:  (N*M, C, T', V)  — e.g. C=256 after the ST-GCN backbone
Output: (N*M, C, T', P)  — P=6 semantic parts (mean pool over joints per part)
"""

import torch
import torch.nn as nn

# NTU RGB+D joint index reference (0-based; official Kinect order in dataset files)
NTU_JOINT_NAMES = (
    'base_spine',      # 0
    'mid_spine',       # 1
    'neck',            # 2
    'head',            # 3
    'left_shoulder',   # 4
    'left_elbow',      # 5
    'left_wrist',      # 6
    'left_hand',       # 7
    'right_shoulder',  # 8
    'right_elbow',     # 9
    'right_wrist',     # 10
    'right_hand',      # 11
    'left_hip',        # 12
    'left_knee',       # 13
    'left_ankle',      # 14
    'left_foot',       # 15
    'right_hip',       # 16
    'right_knee',      # 17
    'right_ankle',     # 18
    'right_foot',      # 19
    'spine_shoulder',  # 20  (graph center; chest / shoulder root)
    'left_hand_tip',   # 21
    'left_thumb',      # 22
    'right_hand_tip',  # 23
    'right_thumb',     # 24
)

# Fixed output order (P=0..5) for KG edges defined in the research plan
PART_NAMES = (
    'head',
    'torso',
    'left_arm',
    'right_arm',
    'left_leg',
    'right_leg',
)

# Disjoint partition of all 25 NTU joints (see module docstring / project notes)
NTU_BODY_PARTS = {
    'head': [2, 3],
    'torso': [0, 1, 20],
    'left_arm': [4, 5, 6, 7, 21, 22],
    'right_arm': [8, 9, 10, 11, 23, 24],
    'left_leg': [12, 13, 14, 15],
    'right_leg': [16, 17, 18, 19],
}

NUM_PARTS = len(PART_NAMES)
NUM_JOINTS_NTU = 25


def _validate_mapping(num_joints=NUM_JOINTS_NTU):
    seen = []
    for name in PART_NAMES:
        for j in NTU_BODY_PARTS[name]:
            if j < 0 or j >= num_joints:
                raise ValueError(
                    'Joint {} in part "{}" out of range [0, {})'.format(
                        j, name, num_joints))
            if j in seen:
                raise ValueError(
                    'Joint {} assigned to more than one part (duplicate in "{}")'.format(
                        j, name))
            seen.append(j)
    if len(seen) != num_joints:
        missing = set(range(num_joints)) - set(seen)
        raise ValueError('Mapping does not cover all joints; missing: {}'.format(
            sorted(missing)))


_validate_mapping()


class BodyPartAggregator(nn.Module):
    """Mean-pool ST-GCN features over semantic body-part joint sets.

    Args:
        part_names: tuple of part keys (default: PART_NAMES order)
        joint_groups: dict part_name -> list of joint indices (default: NTU_BODY_PARTS)

    Shape:
        - Input:  (N, C, T, V)
        - Output: (N, C, T, P)  with P = len(part_names)
    """

    def __init__(self, part_names=PART_NAMES, joint_groups=None):
        super().__init__()
        joint_groups = joint_groups or NTU_BODY_PARTS
        self.part_names = tuple(part_names)
        for name in self.part_names:
            if name not in joint_groups:
                raise KeyError('Missing joint group for part "{}"'.format(name))
        self._part_index_lists = [joint_groups[name] for name in self.part_names]

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) joint-level features (e.g. N = N_batch * M_persons)

        Returns:
            (N, C, T, P) part-level features, P = number of body parts
        """
        if x.dim() != 4:
            raise ValueError(
                'Expected x.ndim == 4 (N, C, T, V), got shape {}'.format(
                    tuple(x.shape)))

        n, c, t, v = x.size()
        if v != NUM_JOINTS_NTU:
            raise ValueError(
                'Expected V={} (NTU RGB+D), got V={}. '
                'Check skeleton layout or provide a custom joint_groups.'.format(
                    NUM_JOINTS_NTU, v))

        # (N, C, T, P): mean over joints in each part; time and channels unchanged
        parts = []
        for joint_idx in self._part_index_lists:
            # x[:, :, :, joint_idx] -> (N, C, T, len(joint_idx))
            part_feat = x[:, :, :, joint_idx].mean(dim=-1)
            parts.append(part_feat.unsqueeze(-1))

        return torch.cat(parts, dim=-1)


if __name__ == '__main__':
    # Minimal sanity check: (N*M, 256, T', 25) -> (N*M, 256, T', 6)
    batch = 4
    channels = 256
    time_steps = 75
    joints = 25

    module = BodyPartAggregator()
    x = torch.randn(batch, channels, time_steps, joints)
    y = module(x)

    assert y.shape == (batch, channels, time_steps, NUM_PARTS), y.shape
    print('Input: ', tuple(x.shape))
    print('Output:', tuple(y.shape))
    print('Parts:', module.part_names)
    print('OK')
