from __future__ import annotations

import math


def cosine_value(initial: float, final: float, progress: float) -> float:
    bounded = min(max(progress, 0.0), 1.0)
    weight = 0.5 * (1.0 + math.cos(math.pi * bounded))
    return final + (initial - final) * weight


def temperature_at_epoch(initial: float, final: float, epoch: int, total_epochs: int) -> float:
    denominator = max(total_epochs - 1, 1)
    return cosine_value(initial, final, epoch / denominator)

