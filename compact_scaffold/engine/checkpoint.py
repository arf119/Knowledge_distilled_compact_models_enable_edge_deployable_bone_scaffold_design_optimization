from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


def random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        seed: int,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "seed": seed,
            "random_state": random_state(),
        }
        torch.save(payload, temporary)
        os.replace(temporary, self.path)

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device | str,
    ) -> tuple[int, int, int]:
        payload = torch.load(self.path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_random_state(payload["random_state"])
        return int(payload["epoch"]), int(payload["step"]), int(payload["seed"])

