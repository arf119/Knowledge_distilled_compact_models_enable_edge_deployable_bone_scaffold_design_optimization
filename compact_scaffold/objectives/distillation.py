from __future__ import annotations

import torch
from torch import Tensor


def distillation_kl(student: Tensor, teacher: Tensor, temperature: float) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError("student and teacher logits must share a shape")
    log_student = torch.log_softmax(student / temperature, dim=-1)
    soft_teacher = torch.softmax(teacher / temperature, dim=-1)
    divergence = torch.nn.functional.kl_div(log_student, soft_teacher, reduction="batchmean")
    return divergence * temperature * temperature

