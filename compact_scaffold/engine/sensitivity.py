from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from compact_scaffold.engine.quantize import PrecisionMap


def hutchinson_trace(
    model: nn.Module,
    loss_closure: Callable[[], Tensor],
    samples: int = 1,
) -> dict[str, float]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    scores = {name: 0.0 for name, _ in parameters}
    for _ in range(samples):
        loss = loss_closure()
        gradients = torch.autograd.grad(
            loss, [parameter for _, parameter in parameters], create_graph=True
        )
        directions = [torch.randn_like(gradient) for gradient in gradients]
        terms = [
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, directions, strict=True)
        ]
        product = torch.stack(terms).sum()
        seconds = torch.autograd.grad(
            product, [parameter for _, parameter in parameters], retain_graph=True
        )
        for (name, _), second, direction in zip(
            parameters, seconds, directions, strict=True
        ):
            scores[name] += float((second * direction).sum().abs().detach()) / samples
    return scores


def choose_precision(
    model: nn.Module,
    sensitivity: dict[str, float],
    fp16_budget: float,
) -> PrecisionMap:
    layers: list[tuple[str, int, float]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            score = sensitivity.get(f"{name}.weight", 0.0)
            layers.append((name, module.weight.numel(), score))
    total = sum(size for _, size, _ in layers)
    budget = int(total * fp16_budget)
    selected: list[str] = []
    occupied = 0
    for name, size, _ in sorted(layers, key=lambda item: item[2], reverse=True):
        pinned = "attention" in name or "readout" in name
        if pinned or occupied + size <= budget:
            selected.append(name)
            occupied += size
    selected_set = set(selected)
    int8 = [name for name, _, _ in layers if name not in selected_set]
    return PrecisionMap(tuple(selected), tuple(int8))
