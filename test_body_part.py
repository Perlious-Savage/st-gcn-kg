#!/usr/bin/env python
"""
Isolated integration check for net/body_part.py against ST-GCN feature layout.

Does NOT import or run the full ST-GCN model — uses a dummy tensor with the
same shape as the backbone output right after the st_gcn_networks loop.

Run from repository root:
    python test_body_part.py
"""

from __future__ import print_function

import os
import sys

# Repository root (this script lives next to main.py / net/)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    # --- ST-GCN tensor shape right after the backbone loop -----------------
    # In net/st_gcn.py, Model.forward:
    #   N, C_in, T_in, V, M = x.size()   # skeleton input
    #   ... reshape to (N * M, C_in, T_in, V)
    #   for gcn, importance in zip(self.st_gcn_networks, ...):
    #       x, _ = gcn(x, self.A * importance)
    #
    # Immediately after that loop, x is:
    #   shape: (N * M, 256, T_prime, V)
    # where:
    #   - First dim = N * M  (batch persons: batch size × max bodies per clip)
    #   - Channel = 256      (final ST-GCN block output width)
    #   - T_prime            (temporal length after two stride-2 temporal convs;
    #                         for common NTU T_in=300, T_prime is typically 75)
    #   - V = 25             (ntu-rgb+d joints; must match Graph.num_node)
    #
    # BodyPartAggregator expects exactly this layout: (N*M, C, T', V).
    # -------------------------------------------------------------------------

    import torch
    from net.body_part import BodyPartAggregator, NUM_PARTS, NUM_JOINTS_NTU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[device]', device)
    if device.type == 'cuda':
        print('[cuda]', torch.cuda.get_device_name(0))

    # Realistic ST-GCN-compatible dimensions (match train.yaml-style batch / NTU M)
    N_batch = 4
    M_bodies = 2
    NM = N_batch * M_bodies
    C_feat = 256
    T_prime = 75  # Typical for T_in=300 after ST-GCN temporal downsampling
    V = NUM_JOINTS_NTU

    x = torch.randn(NM, C_feat, T_prime, V, dtype=torch.float32, device=device)

    print('\n=== Dimension glossary (input to BodyPartAggregator) ===')
    print('  Batch (merged):  N*M = %d  (N_batch=%d × M_persons=%d)' % (NM, N_batch, M_bodies))
    print('  Channels:        C   = %d  (ST-GCN backbone output width)' % C_feat)
    print('  Temporal:        T\'  = %d  (frames after ST-GCN; not raw skeleton T)' % T_prime)
    print('  Joints:          V   = %d  (NTU RGB+D nodes, ST-GCN graph axis)' % V)
    print('  Input shape:     ', tuple(x.shape))

    agg = BodyPartAggregator().to(device)
    y = agg(x)

    print('\n=== After BodyPartAggregator ===')
    print('  Body parts:      P   = %d  (head, torso, arms, legs)' % NUM_PARTS)
    print('  Output shape:    ', tuple(y.shape))

    expected = (NM, C_feat, T_prime, NUM_PARTS)
    assert y.shape == expected, 'Expected %s, got %s' % (expected, tuple(y.shape))
    assert y.device == x.device
    assert torch.isfinite(y).all(), 'Non-finite values in output'

    # Joint axis must be consumed; part axis must be last
    assert y.size(-1) == NUM_PARTS
    assert y.size(-1) != V, 'Output last dim should be parts (6), not joints (25)'

    print('\n[PASS] Shape safety checks OK.')

    # Optional: gradient smoke test (aggregator is pure indexing + mean)
    x_req = torch.randn(NM, C_feat, T_prime, V, device=device, requires_grad=True)
    y_req = agg(x_req)
    y_req.sum().backward()
    assert x_req.grad is not None and torch.isfinite(x_req.grad).all()
    print('[PASS] Autograd backward OK.')


if __name__ == '__main__':
    main()
