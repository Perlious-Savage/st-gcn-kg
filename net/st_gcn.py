import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

_BACKBONE_CHANNELS = 256


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        use_kg (bool): If ``True``, fuse baseline clip features with a
            body-part KG branch. Default ``False`` preserves vanilla ST-GCN.
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, use_kg=False, **kwargs):
        super().__init__()

        # allow use_kg via model_args dict without a dedicated yaml key
        use_kg = kwargs.pop('use_kg', use_kg)
        self.use_kg = use_kg
        self.num_class = num_class

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # Baseline classifier (vanilla ST-GCN): applied after global pool -> (N, 256, 1, 1)
        self.fcn = nn.Conv2d(_BACKBONE_CHANNELS, num_class, kernel_size=1)

        # Optional KG branch (only constructed when enabled; keeps old checkpoints loadable)
        if self.use_kg:
            from net.body_part import BodyPartAggregator
            from net.kg_gnn import KG_GNN
            from net.interaction import DynamicBodyInteraction, TemporalEnergyAttention
            self.body_part = BodyPartAggregator()
            self.kg_gnn = KG_GNN(channels=_BACKBONE_CHANNELS)
            self.dynamic_interaction = DynamicBodyInteraction()
            self.temporal_attention = TemporalEnergyAttention()
            self.fcn_kg = nn.Linear(_BACKBONE_CHANNELS * 2, num_class)

    def _forward_backbone(self, x):
        """Skeleton input -> ST-GCN stack output (joint-level features).

        Returns:
            x: (N*M, 256, T', V) after all st_gcn blocks
            N, M: batch size and max persons (for person pooling)
        """
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        return x, N, M

    def _pool_person(self, x, N, M):
        """Global avg pool over time and nodes, then mean over persons M.

        Args:
            x: (N*M, C, T', S) where S is V (joints) or P (body parts)

        Returns:
            (N, C, 1, 1) clip-level feature map (vanilla layout for self.fcn)
        """
        x = F.avg_pool2d(x, x.size()[2:])
        return x.view(N, M, -1, 1, 1).mean(dim=1)

    def _temporal_weighted_pool(self, x, attn, N, M):
        """Temporal-attention-weighted pool over time and nodes, then mean over persons.

        Args:
            x: (N*M, C, T', S) where S is V or P
            attn: (N*M, T_attn) temporal attention weights
            N, M: batch size and persons

        Returns:
            (N, C) clip-level feature vector
        """
        nm, c, t, s = x.size()
        # Adapt attention length to feature time dimension
        if attn.size(1) != t:
            attn = torch.nn.functional.interpolate(
                attn.unsqueeze(1), size=t, mode='linear', align_corners=False
            ).squeeze(1)
        # attn: (N*M, T') -> (N*M, 1, T', 1) for broadcasting
        w = attn.unsqueeze(1).unsqueeze(-1)  # (N*M, 1, T', 1)
        # Weighted mean over time and spatial dims
        pooled = (x * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) * s)  # (N*M, C)
        pooled = pooled.view(N, M, c).mean(dim=1)  # (N, C)
        return pooled

    def forward(self, x):
        x_raw = x  # save raw skeleton for interaction computation
        # Backbone: (N, C_in, T, V, M) -> (N*M, 256, T', V)
        x, N, M = self._forward_backbone(x)

        if not self.use_kg:
            # --- Vanilla ST-GCN (unchanged) ---
            # (N*M, 256, T', V) -> pool -> (N, 256, 1, 1) -> fcn -> (N, num_class)
            x = self._pool_person(x, N, M)
            x = self.fcn(x)
            return x.view(x.size(0), -1)

        # --- KG-enhanced path ---
        # Compute dynamic interaction adjacency from raw skeleton input
        raw_x = x_raw
        dyn_adj = self.dynamic_interaction(raw_x)  # (N*M, P, P)
        temporal_attn = self.temporal_attention(raw_x)  # (N*M, T-1)

        # Baseline clip embedding from joint features
        base_vec = self._temporal_weighted_pool(x, temporal_attn, N, M)  # (N, 256)

        # Body-part + semantic graph with dynamic adjacency
        part_x = self.body_part(x)              # (N*M, 256, T', 6)
        part_x = self.kg_gnn(part_x, dyn_adj)   # (N*M, 256, T', 6) — dynamic edges

        # KG clip embedding with temporal attention
        kg_vec = self._temporal_weighted_pool(part_x, temporal_attn, N, M)  # (N, 256)

        # Fusion: concat baseline + KG -> (N, 512) -> logits (N, num_class)
        fused = torch.cat([base_vec, kg_vec], dim=1)
        return self.fcn_kg(fused)

    def extract_feature(self, x):
        """Joint-level features and per-(t,v) logits (vanilla layout; KG not applied)."""
        x, N, M = self._forward_backbone(x)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


if __name__ == '__main__':
    # Run from repo root:  python -m net.st_gcn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    num_class = 120
    n, t, v, m = 2, 300, 25, 2
    x = torch.randn(n, 3, t, v, m, device=device)

    for use_kg in (False, True):
        model = Model(
            in_channels=3,
            num_class=num_class,
            graph_args=graph_args,
            edge_importance_weighting=True,
            use_kg=use_kg,
            dropout=0.5,
        ).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (n, num_class), (
            'use_kg={}: expected ({}, {}), got {}'.format(
                use_kg, n, num_class, tuple(out.shape)))
        print('[PASS] use_kg={} output shape {}'.format(use_kg, tuple(out.shape)))