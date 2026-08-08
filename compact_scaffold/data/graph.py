from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CrystalRecord:
    atomic_numbers: Tensor
    positions: Tensor
    cell: Tensor
    target: Tensor
    porosity: float
    family: int

    def validate(self) -> None:
        if self.atomic_numbers.ndim != 1:
            raise ValueError("atomic_numbers must be rank one")
        if self.positions.shape != (self.atomic_numbers.shape[0], 3):
            raise ValueError("positions shape does not match atoms")
        if self.cell.shape != (3, 3):
            raise ValueError("cell must be 3 by 3")
        if not 0.0 <= self.porosity < 1.0:
            raise ValueError("porosity must be in [0, 1)")


def periodic_displacements(positions: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]:
    inverse = torch.linalg.inv(cell)
    fractional = positions @ inverse
    delta = fractional[:, None, :] - fractional[None, :, :]
    image = torch.round(delta)
    wrapped = (delta - image) @ cell
    return wrapped, image.to(torch.int64)


def build_radius_graph(
    positions: Tensor,
    cell: Tensor,
    cutoff: float,
    max_neighbors: int,
) -> tuple[Tensor, Tensor, Tensor]:
    displacement, image = periodic_displacements(positions, cell)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    count = positions.shape[0]
    mask = (distance < cutoff) & (distance > 1.0e-8)
    sources: list[Tensor] = []
    targets: list[Tensor] = []
    shifts: list[Tensor] = []
    for center in range(count):
        candidates = torch.nonzero(mask[center], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        order = torch.argsort(distance[center, candidates])[:max_neighbors]
        chosen = candidates[order]
        sources.append(chosen)
        targets.append(torch.full_like(chosen, center))
        shifts.append(image[center, chosen])
    if not sources:
        empty = torch.empty((2, 0), dtype=torch.int64, device=positions.device)
        vectors = torch.empty((0, 3), device=positions.device)
        radii = torch.empty(0, device=positions.device)
        return empty, vectors, radii
    source = torch.cat(sources)
    target = torch.cat(targets)
    edge_index = torch.stack((source, target))
    shift = torch.cat(shifts)
    vector = positions[source] - positions[target] - shift.to(positions.dtype) @ cell
    radius = torch.linalg.vector_norm(vector, dim=-1)
    return edge_index, vector, radius
