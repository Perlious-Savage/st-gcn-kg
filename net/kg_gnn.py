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

    def __init__(self, channels=256, num_parts=NUM_PARTS, edges=SEMANTIC_EDGES):
        super().__init__()
        self.num_parts = num_parts
        self.channels = channels
        adj = build_part_adjacency(num_parts=num_parts, edges=edges)
        self.register_buffer('adj_norm', adj)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x, dynamic_adj=None):
        """Forward pass with optional dynamic adjacency.

        Args:
            x: Tensor of shape (N, C, T, P).
            dynamic_adj: Optional tensor of shape (N, P, P) providing a
                per-sample adjacency matrix.  When *None* (default), the
                static ``self.adj_norm`` buffer built at init time is used.
        """
        if x.dim() != 4:
            raise ValueError(
                'Expected x.ndim == 4 (N, C, T, P), got shape {}'.format(
                    tuple(x.shape)))

        n, c, t, p = x.size()
        if c != self.channels:
            raise ValueError(
                'Expected C={}, got C={}'.format(self.channels, c))
        if p != self.num_parts:
            raise ValueError(
                'Expected P={}, got P={}'.format(self.num_parts, p))

        # Neighbor messages
        if dynamic_adj is not None:
            # dynamic_adj: (N, P, P) per-sample adjacency
            # x: (N, C, T, P) — flatten C*T, bmm, reshape
            n, c, t, p = x.size()
            x_flat = x.reshape(n, c * t, p)
            agg = torch.bmm(x_flat, dynamic_adj).reshape(n, c, t, p)
        else:
            agg = torch.matmul(x, self.adj_norm)
        return x + self.mix(agg)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[device]', device)

    nm = 8
    c, t, p = 256, 75, NUM_PARTS
    x = torch.randn(nm, c, t, p, device=device)

    print('Input: ', tuple(x.shape))
    gnn = KG_GNN(channels=c).to(device)
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
