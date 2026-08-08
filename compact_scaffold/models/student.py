from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from compact_scaffold.config import DataConfig, ModelConfig
from compact_scaffold.data.collate import CrystalBatch
from compact_scaffold.models.message import InteractionBlock, segment_sum
from compact_scaffold.models.radial import GaussianRadialBasis, SphericalDirections


@dataclass(frozen=True)
class Prediction:
    properties: Tensor
    elastic_tensor: Tensor
    atom_logits: Tensor
    graph_embedding: Tensor


def voigt_tensor(raw: Tensor) -> Tensor:
    diagonal = torch.nn.functional.softplus(raw[:, :6])
    lower = raw.new_zeros((raw.shape[0], 6, 6))
    diagonal_index = torch.arange(6, device=raw.device)
    lower[:, diagonal_index, diagonal_index] = diagonal
    triangle = torch.tril_indices(6, 6, offset=-1, device=raw.device)
    lower[:, triangle[0], triangle[1]] = raw[:, 6:21]
    return lower @ lower.transpose(-1, -2)


class CompactScaffoldNet(nn.Module):
    def __init__(self, model: ModelConfig, data: DataConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(119, model.embedding_dim)
        self.radial = GaussianRadialBasis(model.atomic_features, data.cutoff)
        self.directions = SphericalDirections()
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    model.embedding_dim,
                    model.hidden_dim,
                    model.atomic_features,
                    model.heads,
                    8,
                    model.dropout,
                )
                for _ in range(model.layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(model.embedding_dim, model.hidden_dim),
            nn.SiLU(),
            nn.Linear(model.hidden_dim, model.outputs),
        )
        self.atom_head = nn.Linear(model.embedding_dim, model.outputs)

    def forward(self, batch: CrystalBatch) -> Prediction:
        nodes = self.embedding(batch.atomic_numbers)
        radial = self.radial(batch.edge_lengths)
        directions = self.directions(batch.edge_vectors)
        for interaction in self.interactions:
            nodes = interaction(nodes, batch.edge_index, radial, directions)
        graph_sum = segment_sum(nodes, batch.graph_index, batch.graphs)
        graph_count = torch.bincount(batch.graph_index, minlength=batch.graphs).clamp_min(1)
        graph_embedding = graph_sum / graph_count[:, None]
        output = self.readout(graph_embedding)
        elastic = voigt_tensor(output[:, 3:24])
        return Prediction(output[:, :3], elastic, self.atom_head(nodes), graph_embedding)
