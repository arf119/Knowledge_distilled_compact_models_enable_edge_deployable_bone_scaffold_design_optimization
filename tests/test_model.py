import torch

from compact_scaffold.config import ExperimentConfig
from compact_scaffold.data.collate import CrystalBatch
from compact_scaffold.models.student import CompactScaffoldNet


def test_model_outputs_are_finite(config: ExperimentConfig, batch: CrystalBatch) -> None:
    model = CompactScaffoldNet(config.model, config.data)
    output = model(batch)
    assert output.properties.shape == (2, 3)
    assert output.elastic_tensor.shape == (2, 6, 6)
    assert output.atom_logits.shape == (8, 24)
    assert torch.isfinite(output.properties).all()


def test_elastic_tensor_is_symmetric(config: ExperimentConfig, batch: CrystalBatch) -> None:
    model = CompactScaffoldNet(config.model, config.data)
    elastic = model(batch).elastic_tensor
    assert torch.allclose(elastic, elastic.transpose(-1, -2))

