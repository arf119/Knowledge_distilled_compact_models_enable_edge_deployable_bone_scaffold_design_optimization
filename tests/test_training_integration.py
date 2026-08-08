import torch

from compact_scaffold.config import ExperimentConfig
from compact_scaffold.data.collate import CrystalBatch
from compact_scaffold.engine.trainer import Trainer
from compact_scaffold.models.student import CompactScaffoldNet
from compact_scaffold.objectives.total import JointLoss


def test_two_training_updates(config: ExperimentConfig, batch: CrystalBatch) -> None:
    torch.manual_seed(config.seed)
    model = CompactScaffoldNet(config.model, config.data)
    trainer = Trainer(model, JointLoss(config.loss), config, torch.device("cpu"))
    teacher = batch.targets[:, :3]
    first = trainer.update(batch, teacher, 0).total.item()
    second = trainer.update(batch, teacher, 0).total.item()
    assert torch.isfinite(torch.tensor([first, second])).all()
    assert trainer.step == 2
