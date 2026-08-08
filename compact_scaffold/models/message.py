from __future__ import annotations

import torch
from torch import Tensor, nn


def segment_sum(values: Tensor, index: Tensor, count: int) -> Tensor:
    shape = (count,) + values.shape[1:]
    result = values.new_zeros(shape)
    result.index_add_(0, index, values)
    return result


class GatedMessage(nn.Module):
    def __init__(self, dimension: int, radial_channels: int) -> None:
        super().__init__()
        self.source = nn.Linear(dimension, dimension, bias=False)
        self.radial = nn.Sequential(
            nn.Linear(radial_channels, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.gate = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
            nn.Sigmoid(),
        )

    def forward(self, nodes: Tensor, edge_index: Tensor, radial: Tensor) -> Tensor:
        source, target = edge_index
        transformed = self.source(nodes[source]) * self.radial(radial)
        gate = self.gate(torch.cat((nodes[source], nodes[target]), dim=-1))
        return segment_sum(transformed * gate, target, nodes.shape[0])


class EquivariantAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, direction_channels: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must divide evenly across heads")
        self.heads = heads
        self.width = dimension // heads
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.direction = nn.Linear(direction_channels, heads, bias=False)
        self.output = nn.Linear(dimension, dimension)

    def forward(self, nodes: Tensor, edge_index: Tensor, directions: Tensor) -> Tensor:
        source, target = edge_index
        query = self.query(nodes[target]).view(-1, self.heads, self.width)
        key = self.key(nodes[source]).view(-1, self.heads, self.width)
        value = self.value(nodes[source]).view(-1, self.heads, self.width)
        logits = (query * key).sum(-1) / self.width**0.5
        logits = logits + self.direction(directions)
        weights = torch.sigmoid(logits)
        message = value * weights[..., None]
        reduced = segment_sum(message, target, nodes.shape[0])
        output: Tensor = self.output(reduced.flatten(1))
        return output


class InteractionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden: int,
        radial_channels: int,
        heads: int,
        direction_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.message = GatedMessage(dimension, radial_channels)
        self.attention = EquivariantAttention(dimension, heads, direction_channels)
        self.norm_one = nn.LayerNorm(dimension)
        self.norm_two = nn.LayerNorm(dimension)
        self.update = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
        )

    def forward(
        self,
        nodes: Tensor,
        edge_index: Tensor,
        radial: Tensor,
        directions: Tensor,
    ) -> Tensor:
        messages = self.message(nodes, edge_index, radial)
        attention = self.attention(nodes, edge_index, directions)
        nodes = self.norm_one(nodes + messages + attention)
        output: Tensor = self.norm_two(nodes + self.update(nodes))
        return output
