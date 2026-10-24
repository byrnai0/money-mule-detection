
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPLayer(nn.Module):
    """CP-inspired multiplicative pooling over graph neighbors."""

    def __init__(self, in_channels, out_channels, rank=16):
        super().__init__()

        if rank < 1:
            raise ValueError("rank must be >= 1")

        self.W = nn.Parameter(
            torch.empty(in_channels + 1, rank)
        )
        self.M = nn.Linear(rank, out_channels, bias=False)

        nn.init.xavier_uniform_(self.W)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        source, target = edge_index

        # Append homogeneous coordinate 1 to each node feature.
        x_augmented = torch.cat(
            [x, x.new_ones((num_nodes, 1))], dim=1
        )

        # Shared projection into the low-rank CP space.
        projected = torch.tanh(x_augmented @ self.W)

        # Multiply projected representations of each node's
        # incoming neighbors. Nodes with no incoming edges
        # retain the multiplicative identity (all ones).
        neighbor_product = x.new_ones(
            (num_nodes, self.W.size(1))
        )

        if target.numel() > 0:
            indices = target.unsqueeze(1).expand(
                -1, self.W.size(1)
            )
            neighbor_product.scatter_reduce_(
                dim=0,
                index=indices,
                src=projected[source],
                reduce="prod",
                include_self=True,
            )

        # No activation here; the caller controls hidden/final use.
        return self.M(neighbor_product)


class CPPoolingGNN(nn.Module):
    """Two-layer CP pooling network for node classification."""

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=2,
        rank=16,
        dropout=0.30,
    ):
        super().__init__()

        self.cp1 = CPLayer(
            in_channels, hidden_channels, rank
        )
        self.cp2 = CPLayer(
            hidden_channels, out_channels, rank
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.cp1(x, edge_index))
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # Return unrestricted logits for CrossEntropyLoss.
        return self.cp2(x, edge_index)
