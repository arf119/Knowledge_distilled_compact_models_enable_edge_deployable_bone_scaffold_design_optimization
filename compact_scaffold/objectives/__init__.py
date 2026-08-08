from compact_scaffold.objectives.distillation import distillation_kl
from compact_scaffold.objectives.physics import gibson_ashby_loss, spd_loss
from compact_scaffold.objectives.total import JointLoss, LossTerms

__all__ = ["JointLoss", "LossTerms", "distillation_kl", "gibson_ashby_loss", "spd_loss"]

