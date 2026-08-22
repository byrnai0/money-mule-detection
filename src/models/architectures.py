from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    ChebConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
)


class GCNModel(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


class GATModel(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


class GATv2Model(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


class GraphSAGEModel(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


class GINModel(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(
                in_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
        )

        mlp2 = nn.Sequential(
            nn.Linear(
                hidden_channels,
                hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_channels,
                out_channels,
            ),
        )

        self.conv1 = GINConv(
            mlp1
        )

        self.conv2 = GINConv(
            mlp2
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


class ChebModel(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        k: int = 3,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.conv1 = ChebConv(
            in_channels,
            hidden_channels,
            K=k,
        )

        self.conv2 = ChebConv(
            hidden_channels,
            out_channels,
            K=k,
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(x, edge_index)

        return x


MODEL_REGISTRY = {
    "gcn": GCNModel,
    "gat": GATModel,
    "gatv2": GATv2Model,
    "sage": GraphSAGEModel,
    "gin": GINModel,
    "cheb": ChebModel,
}


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    dropout: float = 0.30,
) -> nn.Module:

    name = name.lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    model_class = MODEL_REGISTRY[name]

    return model_class(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )