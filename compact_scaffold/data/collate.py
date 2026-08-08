from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from compact_scaffold.data.graph import CrystalRecord, build_radius_graph


@dataclass(frozen=True)
class CrystalBatch:
    atomic_numbers: Tensor
    positions: Tensor
    edge_index: Tensor
    edge_vectors: Tensor
    edge_lengths: Tensor
    graph_index: Tensor
    targets: Tensor
    porosity: Tensor
    family: Tensor
    cells: Tensor

    @property
    def graphs(self) -> int:
        return int(self.targets.shape[0])

    def to(self, device: torch.device | str) -> CrystalBatch:
        values = {name: value.to(device) for name, value in self.__dict__.items()}
        return CrystalBatch(**values)


def collate_crystals(
    records: list[CrystalRecord],
    cutoff: float,
    max_neighbors: int,
) -> CrystalBatch:
    if not records:
        raise ValueError("cannot collate an empty sequence")
    atomic_numbers: list[Tensor] = []
    positions: list[Tensor] = []
    edges: list[Tensor] = []
    vectors: list[Tensor] = []
    lengths: list[Tensor] = []
    graphs: list[Tensor] = []
    offset = 0
    for graph_id, record in enumerate(records):
        record.validate()
        edge_index, edge_vector, edge_length = build_radius_graph(
            record.positions, record.cell, cutoff, max_neighbors
        )
        atomic_numbers.append(record.atomic_numbers)
        positions.append(record.positions)
        edges.append(edge_index + offset)
        vectors.append(edge_vector)
        lengths.append(edge_length)
        graphs.append(torch.full_like(record.atomic_numbers, graph_id))
        offset += record.atomic_numbers.shape[0]
    return CrystalBatch(
        atomic_numbers=torch.cat(atomic_numbers),
        positions=torch.cat(positions),
        edge_index=torch.cat(edges, dim=1),
        edge_vectors=torch.cat(vectors),
        edge_lengths=torch.cat(lengths),
        graph_index=torch.cat(graphs),
        targets=torch.stack([record.target for record in records]),
        porosity=torch.tensor([record.porosity for record in records], dtype=torch.float32),
        family=torch.tensor([record.family for record in records], dtype=torch.int64),
        cells=torch.stack([record.cell for record in records]),
    )
