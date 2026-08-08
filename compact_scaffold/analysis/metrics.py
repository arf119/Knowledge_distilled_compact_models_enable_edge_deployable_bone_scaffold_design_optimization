from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RegressionSummary:
    mae: float
    rmse: float
    r2: float


def mean_absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).abs().mean()


def root_mean_square_error(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt((prediction - target).square().mean())


def coefficient_of_determination(prediction: Tensor, target: Tensor) -> Tensor:
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum().clamp_min(1.0e-12)
    return 1.0 - residual / total


def regression_summary(prediction: Tensor, target: Tensor) -> RegressionSummary:
    return RegressionSummary(
        float(mean_absolute_error(prediction, target)),
        float(root_mean_square_error(prediction, target)),
        float(coefficient_of_determination(prediction, target)),
    )


def spd_violation_rate(tensors: Tensor) -> Tensor:
    smallest = torch.linalg.eigvalsh(tensors)[:, 0]
    result: Tensor = (smallest < 0.0).to(torch.float32).mean()
    return result


def rotation_equivariance_error(reference: Tensor, rotated: Tensor) -> Tensor:
    denominator = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1.0e-8)
    numerator = torch.linalg.vector_norm(reference - rotated, dim=-1)
    result: Tensor = (numerator / denominator).amax()
    return result
