from __future__ import annotations

import torch
from torch import Tensor


def spd_loss(elastic_tensor: Tensor) -> Tensor:
    eigenvalues = torch.linalg.eigvalsh(elastic_tensor)
    violation = torch.relu(-eigenvalues[:, 0])
    return violation.square().mean()


def gibson_ashby_prediction(
    dense_modulus: Tensor,
    porosity: Tensor,
    family: Tensor,
    exponents: Tensor,
    coefficient: Tensor | None = None,
) -> Tensor:
    scale = torch.ones_like(dense_modulus) if coefficient is None else coefficient
    relative_density = (1.0 - porosity).clamp_min(1.0e-6)
    return scale * dense_modulus * relative_density.pow(exponents[family])


def gibson_ashby_loss(
    effective_modulus: Tensor,
    dense_modulus: Tensor,
    porosity: Tensor,
    family: Tensor,
    exponents: Tensor,
) -> Tensor:
    expected = gibson_ashby_prediction(dense_modulus, porosity, family, exponents)
    residual = torch.log(effective_modulus.clamp_min(1.0e-6)) - torch.log(
        expected.clamp_min(1.0e-6)
    )
    return residual.square().mean()

