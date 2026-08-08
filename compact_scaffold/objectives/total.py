from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from compact_scaffold.config import LossConfig
from compact_scaffold.data.collate import CrystalBatch
from compact_scaffold.models.student import Prediction
from compact_scaffold.objectives.distillation import distillation_kl
from compact_scaffold.objectives.physics import gibson_ashby_loss, spd_loss


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    task: Tensor
    distillation: Tensor
    spd: Tensor
    gibson_ashby: Tensor


class JointLoss(nn.Module):
    exponents: Tensor

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.register_buffer(
            "exponents",
            torch.tensor(
                [config.hap_exponent, config.tcp_exponent, config.glass_exponent],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        prediction: Prediction,
        batch: CrystalBatch,
        teacher_logits: Tensor,
        temperature: float,
    ) -> LossTerms:
        task = torch.nn.functional.l1_loss(prediction.properties, batch.targets[:, :3])
        teacher_atoms = teacher_logits[batch.graph_index]
        kd = distillation_kl(prediction.atom_logits[:, :3], teacher_atoms[:, :3], temperature)
        spd = spd_loss(prediction.elastic_tensor)
        ga = gibson_ashby_loss(
            prediction.properties[:, 1],
            batch.targets[:, 1],
            batch.porosity,
            batch.family,
            self.exponents,
        )
        total = self.gamma * task + self.alpha * kd + self.beta * (spd + ga)
        return LossTerms(total, task, kd, spd, ga)
