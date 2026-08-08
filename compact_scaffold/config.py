from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    cutoff: float
    max_neighbors: int
    max_atoms: int
    batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    atomic_features: int
    embedding_dim: int
    hidden_dim: int
    layers: int
    heads: int
    outputs: int
    dropout: float


@dataclass(frozen=True)
class LossConfig:
    alpha: float
    beta: float
    gamma: float
    temperature_initial: float
    temperature_final: float
    hap_exponent: float
    tcp_exponent: float
    glass_exponent: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    warmup_epochs: int
    full_precision_epoch: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    precision: str
    world_size: int
    gradient_accumulation: int


@dataclass(frozen=True)
class QuantizationConfig:
    fp16_budget: float
    bits: int
    per_channel: bool
    symmetric: bool


@dataclass(frozen=True)
class SearchConfig:
    population: int
    budget_seconds: float
    porosity_min: float
    porosity_max: float
    patience: int
    tolerance: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    quantization: QuantizationConfig
    search: SearchConfig

    @property
    def effective_batch_size(self) -> int:
        return self.data.batch_size * self.train.world_size * self.train.gradient_accumulation


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {name} must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    return ExperimentConfig(
        seed=int(raw["seed"]),
        data=DataConfig(**_section(raw, "data")),
        model=ModelConfig(**_section(raw, "model")),
        loss=LossConfig(**_section(raw, "loss")),
        train=TrainConfig(**_section(raw, "train")),
        quantization=QuantizationConfig(**_section(raw, "quantization")),
        search=SearchConfig(**_section(raw, "search")),
    )

