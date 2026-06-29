"""
Lightweight semantic graph reasoning over body-part features.

Input:  (N, C, T, P)  — e.g. output of BodyPartAggregator, C=256, P=6
Output: (N, C, T, P)  — same shape; one residual message-passing step

Graph (undirected, torso-centered), part order matches net.body_part.PART_NAMES:
  0 head, 1 torso, 2 left_arm, 3 right_arm, 4 left_leg, 5 right_leg
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from net.body_part import PART_NAMES, NUM_PARTS

# Semantic edges (body-part names)
SEMANTIC_EDGES = (
    ('head', 'torso'),
    ('torso', 'left_arm'),
    ('torso', 'right_arm'),
    ('torso', 'left_leg'),
    ('torso', 'right_leg'),
)

_PART_INDEX = {name: i for i, name in enumerate(PART_NAMES)}


def _edges_to_index_pairs(edges):
    pairs = []
    for a, b in edges:
        i, j = _PART_INDEX[a], _PART_INDEX[b]
        pairs.append((i, j))
    return pairs


def build_part_adjacency(num_parts=NUM_PARTS, edges=SEMANTIC_EDGES, self_loop=True):
    """Symmetric adjacency with optional self-loops, row-normalized for aggregation."""
    adj = torch.zeros(num_parts, num_parts, dtype=torch.float32)
    for a, b in _edges_to_index_pairs(edges):
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    if self_loop:
        adj = adj + torch.eye(num_parts, dtype=torch.float32)
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


class KG_GNN(nn.Module):
    """One-step neighbor aggregation on the body-part graph with a residual update.

    For each time step and channel, part features are mixed along the graph:
        agg = A_norm @ x
        out = x + Conv2d(1x1)(agg)

    Shape:
        - Input:  (N, C, T, P)
        - Output: (N, C, T, P)
    """

    def __init__(self, channels=256, num_parts=NUM_PARTS, edges=SEMANTIC_EDGES, use_dynamic_adj=False, dynamic_alpha=0.5, clamp_nonnegative=False, use_column_norm=False):
        super().__init__()
        self.num_parts = num_parts
        self.channels = channels
        self.use_dynamic_adj = use_dynamic_adj
        self.dynamic_alpha = dynamic_alpha
        self.clamp_nonnegative = clamp_nonnegative
        self.use_column_norm = use_column_norm
        adj = build_part_adjacency(num_parts=num_parts, edges=edges)
        self.register_buffer('adj_norm', adj)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        n, c, t, p = x.size()
        if c != self.channels:
            raise ValueError(
                'Expected C={}, got C={}'.format(self.channels, c))
        if p != self.num_parts:
            raise ValueError(
                'Expected P={}, got P={}'.format(self.num_parts, p))

        if self.use_dynamic_adj:
            # 1. Flatten spatial-temporal channels to compute correlation over (C * T) features
            # part_flat shape: (N, T * C, P)
            part_flat = x.permute(0, 2, 1, 3).contiguous().view(n, t * c, p)
            part_mean = part_flat.mean(dim=1, keepdim=True)
            part_centered = part_flat - part_mean
            part_std = torch.sqrt(part_centered.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            part_norm = part_centered / part_std
            # Batch matrix multiplication: (N, P, T*C) @ (N, T*C, P) -> (N, P, P)
            adj_dyn = torch.bmm(part_norm.transpose(1, 2), part_norm)

            if self.clamp_nonnegative:
                adj_dyn = torch.clamp(adj_dyn, min=0.0)
            
            # 2. Fuse with static adjacency matrix:
            # self.adj_norm shape is (P, P). Broadcasts to (N, P, P).
            adj_fused = (1.0 - self.dynamic_alpha) * self.adj_norm + self.dynamic_alpha * adj_dyn
            
            # Row or Column normalization to keep feature scales stable:
            if self.use_column_norm:
                deg = adj_fused.abs().sum(dim=1, keepdim=True).clamp(min=1.0)
                adj_fused = adj_fused / deg
            else:
                deg = adj_fused.abs().sum(dim=2, keepdim=True).clamp(min=1.0)
                adj_fused = adj_fused / deg
            
            if getattr(self, '_first_pass', True):
                self._first_pass = False
                print("\n--- DEBUG: Dynamic Adjacency (First Pass) ---")
                print("Shape: {}".format(tuple(adj_dyn.shape)))
                print("Mean: {:.6f}".format(adj_dyn.mean().item()))
                print("Std: {:.6f}".format(adj_dyn.std().item()))
                print("Min: {:.6f}".format(adj_dyn.min().item()))
                print("Max: {:.6f}".format(adj_dyn.max().item()))
                print("Row sums (sample 0): {}".format(adj_fused[0].abs().sum(dim=1).tolist()))
                print("Column sums (sample 0): {}".format(adj_fused[0].abs().sum(dim=0).tolist()))
                diag_mask = torch.eye(adj_dyn.size(-1), device=adj_dyn.device).bool()
                diag_vals = adj_dyn[:, diag_mask]
                off_diag_vals = adj_dyn[:, ~diag_mask]
                print("Diagonal mean: {:.6f}".format(diag_vals.mean().item()))
                print("Off-diagonal mean: {:.6f}".format(off_diag_vals.mean().item()))
                print("---------------------------------------------\n")
            
            # 3. Message passing: (N, C, T, P) -> permute to (N, C*T, P)
            # (N, C*T, P) @ (N, P, P) -> (N, C*T, P) -> view back to (N, C, T, P)
            x_flat = x.permute(0, 1, 2, 3).contiguous().view(n, c * t, p)
            agg = torch.bmm(x_flat, adj_fused).view(n, c, t, p)
        else:
            # Neighbor messages: (N, C, T, P) @ (P, P) -> (N, C, T, P)
            agg = torch.matmul(x, self.adj_norm)

        return x + self.mix(agg)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[device]', device)

    nm = 8
    c, t, p = 256, 75, NUM_PARTS
    x = torch.randn(nm, c, t, p, device=device)

    print('Input: ', tuple(x.shape))
    for use_dyn in (False, True):
        print('Testing KG_GNN with use_dynamic_adj={}'.format(use_dyn))
        gnn = KG_GNN(channels=c, use_dynamic_adj=use_dyn, dynamic_alpha=0.5).to(device)
        y = gnn(x)
        print('Output:', tuple(y.shape))

        assert y.shape == x.shape
        assert y.device == x.device
        assert torch.isfinite(y).all()

        x_req = x.detach().clone().requires_grad_(True)
        y_req = gnn(x_req)
        y_req.sum().backward()
        assert x_req.grad is not None and torch.isfinite(x_req.grad).all()

    print('[PASS] kg_gnn shape, device, autograd OK')
