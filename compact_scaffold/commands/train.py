from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from compact_scaffold.config import load_config
from compact_scaffold.models.student import CompactScaffoldNet
from compact_scaffold.objectives.total import JointLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="compact-scaffold-train")
    value.add_argument("--config", type=Path, default=Path("settings/main.yaml"))
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)
    set_seed(config.seed)
    device = torch.device(arguments.device)
    model = CompactScaffoldNet(config.model, config.data).to(device)
    objective = JointLoss(config.loss).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    logging.info(
        "model_parameters=%d effective_batch=%d device=%s objective=%s",
        parameters,
        config.effective_batch_size,
        device,
        objective.__class__.__name__,
    )


if __name__ == "__main__":
    main()

