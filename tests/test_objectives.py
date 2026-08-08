import torch

from compact_scaffold.objectives.distillation import distillation_kl
from compact_scaffold.objectives.physics import gibson_ashby_prediction, spd_loss


def test_distillation_identity_is_zero() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(distillation_kl(logits, logits, 4.0), torch.tensor(0.0), atol=1.0e-6)


def test_spd_loss_detects_negative_eigenvalue() -> None:
    tensor = torch.eye(6).unsqueeze(0)
    assert spd_loss(tensor) == 0.0
    tensor[:, 0, 0] = -2.0
    assert spd_loss(tensor) == 4.0


def test_gibson_ashby_family_exponent() -> None:
    value = gibson_ashby_prediction(
        torch.tensor([10.0]),
        torch.tensor([0.5]),
        torch.tensor([0]),
        torch.tensor([2.0, 1.9, 1.8]),
    )
    assert torch.allclose(value, torch.tensor([2.5]))

