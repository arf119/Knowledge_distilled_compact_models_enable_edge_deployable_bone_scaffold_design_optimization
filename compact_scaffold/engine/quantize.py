from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def symmetric_fake_quantize(weight: Tensor, bits: int = 8) -> Tensor:
    ceiling = 2 ** (bits - 1) - 1
    reduce = tuple(range(1, weight.ndim))
    scale = weight.detach().abs().amax(dim=reduce, keepdim=True).clamp_min(1.0e-8) / ceiling
    integer = torch.round(weight / scale).clamp(-ceiling, ceiling)
    quantized = integer * scale
    result: Tensor = weight + (quantized - weight).detach()
    return result


class QuantizedLinear(nn.Linear):
    bits: int

    def __init__(self, source: nn.Linear, bits: int = 8) -> None:
        super().__init__(source.in_features, source.out_features, source.bias is not None)
        self.weight = source.weight
        self.bias = source.bias
        self.bits = bits

    def forward(self, values: Tensor) -> Tensor:
        weight = symmetric_fake_quantize(self.weight, self.bits)
        return torch.nn.functional.linear(values, weight, self.bias)


@dataclass(frozen=True)
class PrecisionMap:
    fp16: tuple[str, ...]
    int8: tuple[str, ...]


def attach_fake_quantization(model: nn.Module, precision: PrecisionMap, bits: int = 8) -> None:
    modules = dict(model.named_modules())
    for name in precision.int8:
        module = modules[name]
        if not isinstance(module, nn.Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child_name, QuantizedLinear(module, bits))
