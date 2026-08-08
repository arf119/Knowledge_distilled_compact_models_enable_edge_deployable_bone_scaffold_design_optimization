from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from compact_scaffold.data.collate import CrystalBatch
from compact_scaffold.models.message import segment_sum
from compact_scaffold.models.student import Prediction


class TeacherModel(Protocol):
    def __call__(self, batch: CrystalBatch) -> Prediction: ...


@dataclass(frozen=True)
class TeacherTargets:
    properties: Tensor
    atom_logits: Tensor
    gate: Tensor


class PhysicsOracle(nn.Module):
    exponents: Tensor

    def __init__(self, exponents: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("exponents", torch.tensor(exponents))

    def forward(self, batch: CrystalBatch, dense_modulus: Tensor) -> Tensor:
        exponent = self.exponents[batch.family]
        relative_density = 1.0 - batch.porosity
        effective = dense_modulus * relative_density.pow(exponent)
        shear = effective * 0.4
        formation = batch.targets[:, 0]
        return torch.stack((formation, effective, shear), dim=-1)


class DualTeacher(nn.Module):
    def __init__(self, dimension: int, outputs: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dimension, 1), nn.Sigmoid())
        self.outputs = outputs

    def forward(
        self,
        batch: CrystalBatch,
        atom_features: Tensor,
        first: Tensor,
        second: Tensor,
    ) -> TeacherTargets:
        if first.shape != second.shape:
            raise ValueError("teacher logits must have identical shapes")
        atom_gate = self.gate(atom_features)
        graph_gate_sum = segment_sum(atom_gate, batch.graph_index, batch.graphs)
        counts = torch.bincount(batch.graph_index, minlength=batch.graphs).clamp_min(1)
        graph_gate = graph_gate_sum / counts[:, None]
        fused = graph_gate * first + (1.0 - graph_gate) * second
        expanded = fused[batch.graph_index]
        return TeacherTargets(fused[:, :3], expanded[:, : self.outputs], graph_gate)
