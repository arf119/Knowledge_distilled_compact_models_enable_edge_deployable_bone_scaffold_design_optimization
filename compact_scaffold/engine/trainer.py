from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from compact_scaffold.config import ExperimentConfig
from compact_scaffold.data.collate import CrystalBatch
from compact_scaffold.engine.schedule import temperature_at_epoch
from compact_scaffold.models.student import CompactScaffoldNet
from compact_scaffold.objectives.total import JointLoss, LossTerms


@dataclass(frozen=True)
class EpochStatistics:
    loss: float
    task: float
    distillation: float
    physics: float
    batches: int


class Trainer:
    def __init__(
        self,
        model: CompactScaffoldNet,
        objective: JointLoss,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.train.epochs
        )
        self.logger = logging.getLogger(__name__)
        self.step = 0

    def _autocast(self) -> contextlib.AbstractContextManager[None]:
        enabled = self.device.type == "cuda" and self.config.train.precision in {"bf16", "fp16"}
        dtype = torch.bfloat16 if self.config.train.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def update(self, batch: CrystalBatch, teacher_logits: Tensor, epoch: int) -> LossTerms:
        self.model.train()
        batch = batch.to(self.device)
        teacher_logits = teacher_logits.to(self.device)
        temperature = temperature_at_epoch(
            self.config.loss.temperature_initial,
            self.config.loss.temperature_final,
            epoch,
            self.config.train.epochs,
        )
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            prediction = self.model(batch)
            terms = self.objective(prediction, batch, teacher_logits, temperature)
        terms.total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip)
        self.optimizer.step()
        self.step += 1
        result: LossTerms = terms
        return result

    def train_epoch(
        self,
        batches: Iterable[tuple[CrystalBatch, Tensor]],
        epoch: int,
    ) -> EpochStatistics:
        totals = torch.zeros(4)
        count = 0
        for batch, teacher in batches:
            terms = self.update(batch, teacher, epoch)
            totals += torch.tensor(
                [
                    terms.total.detach().cpu(),
                    terms.task.detach().cpu(),
                    terms.distillation.detach().cpu(),
                    (terms.spd + terms.gibson_ashby).detach().cpu(),
                ]
            )
            count += 1
        if count == 0:
            raise ValueError("training epoch received no batches")
        self.scheduler.step()
        means = totals / count
        result = EpochStatistics(
            loss=float(means[0]),
            task=float(means[1]),
            distillation=float(means[2]),
            physics=float(means[3]),
            batches=count,
        )
        self.logger.info("epoch=%d loss=%.6f batches=%d", epoch, result.loss, count)
        return result
