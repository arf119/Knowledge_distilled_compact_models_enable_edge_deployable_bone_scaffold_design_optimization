import torch
from torch import nn

from compact_scaffold.engine.quantize import (
    PrecisionMap,
    attach_fake_quantization,
    symmetric_fake_quantize,
)


def test_fake_quant_preserves_gradient() -> None:
    value = torch.randn(4, 3, requires_grad=True)
    symmetric_fake_quantize(value).sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_precision_map_replaces_linear() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    attach_fake_quantization(model, PrecisionMap((), ("0", "2")))
    output = model(torch.randn(2, 3))
    assert output.shape == (2, 2)
