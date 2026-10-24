from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPInspiredAggregation(nn.Module):
    """
    Low-rank, CP-inspired node-neighbor interaction layer.

    For each node:
      1. Aggregate neighboring node representations by mean.
      2. Project the node and neighbor representations into a
         shared low-rank space.
      3. Multiply the projections element-wise.
      4. Project the interaction into the output space.

    This is a CP-inspired factorized interaction, not a claim to
    reproduce a particular published CP-GCN implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 16,
    ):
        super().__init__()

        if rank < 1:
            raise ValueError("rank must be at least 1")

        self.node_projection = nn.Linear(
            in_channels, rank, bias=False
        )
        self.neighbor_projection = nn.Linear(
            in_channels, rank, bias=False
        )
        self.output_projection = nn.Linear(
            rank, out_channels
        )

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        source, target = edge_index

        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, target, x[source])

        degree = x.new_zeros(num_nodes)
        degree.index_add_(
            0, target, x.new_ones(target.numel())
        )

        neighbor_mean = neighbor_sum / degree.clamp_min(1).unsqueeze(-1)

        node_factor = self.node_projection(x)
        neighbor_factor = self.neighbor_projection(neighbor_mean)

        interaction = node_factor * neighbor_factor
        return self.output_projection(interaction)


class CPGNNModel(nn.Module):
    """Two-layer CP-inspired GNN for node classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        rank: int = 16,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = CPInspiredAggregation(
            in_channels, hidden_channels, rank
        )
        self.conv2 = CPInspiredAggregation(
            hidden_channels, out_channels, rank
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(
            x, p=self.dropout, training=self.training
        )
        return self.conv2(x, edge_index)
