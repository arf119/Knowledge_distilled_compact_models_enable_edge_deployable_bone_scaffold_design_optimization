from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SmoothCutoff(nn.Module):
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff

    def forward(self, radius: Tensor) -> Tensor:
        ratio = radius / self.cutoff
        envelope = 0.5 * (torch.cos(math.pi * ratio) + 1.0)
        return torch.where(radius < self.cutoff, envelope, torch.zeros_like(envelope))


class GaussianRadialBasis(nn.Module):
    centers: Tensor

    def __init__(self, channels: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, channels)
        spacing = cutoff / max(channels - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(spacing * spacing, 1.0e-8)
        self.envelope = SmoothCutoff(cutoff)

    def forward(self, radius: Tensor) -> Tensor:
        delta = radius[:, None] - self.centers[None, :]
        result: Tensor = torch.exp(-self.gamma * delta.square()) * self.envelope(radius)[:, None]
        return result


class SphericalDirections(nn.Module):
    def forward(self, vectors: Tensor) -> Tensor:
        radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1.0e-8)
        direction = vectors / radius
        x, y, z = direction.unbind(-1)
        second = torch.stack(
            (
                x * y,
                y * z,
                z * x,
                x.square() - y.square(),
                3.0 * z.square() - 1.0,
            ),
            dim=-1,
        )
        return torch.cat((direction, second), dim=-1)
