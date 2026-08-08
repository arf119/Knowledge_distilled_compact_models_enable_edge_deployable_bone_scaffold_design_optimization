from __future__ import annotations

from pathlib import Path

import pytest
import torch

from compact_scaffold.config import ExperimentConfig, load_config
from compact_scaffold.data.collate import CrystalBatch, collate_crystals
from compact_scaffold.data.graph import CrystalRecord


@pytest.fixture
def config() -> ExperimentConfig:
    return load_config(Path(__file__).parents[1] / "settings" / "test.yaml")


@pytest.fixture
def batch(config: ExperimentConfig) -> CrystalBatch:
    records = []
    for offset in (0.0, 0.1):
        records.append(
            CrystalRecord(
                atomic_numbers=torch.tensor([8, 20, 15, 8]),
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]
                )
                + offset,
                cell=torch.eye(3) * 8.0,
                target=torch.tensor([-1.2, 15.0, 6.0]),
                porosity=0.45,
                family=0,
            )
        )
    return collate_crystals(records, config.data.cutoff, config.data.max_neighbors)

