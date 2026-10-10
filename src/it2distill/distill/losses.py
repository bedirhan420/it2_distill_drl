"""Loss Strategy: chosen from ActionSpec/config, not if-chains in the trainer."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from it2distill.core.registry import Registry
from it2distill.core.types import ActionSpec

LOSS_REGISTRY: Registry["DistillLoss"] = Registry("loss")


class DistillLoss(ABC):
    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropy(DistillLoss):
    def __init__(self, **_) -> None: ...
    def __call__(self, pred, target):
        return F.cross_entropy(pred, target)


@LOSS_REGISTRY.register("mse")
class MSE(DistillLoss):
    def __init__(self, **_) -> None: ...
    def __call__(self, pred, target):
        return F.mse_loss(pred, target)


@LOSS_REGISTRY.register("kl_soft")
class SoftKL(DistillLoss):
    """Hinton-style: target = teacher probabilities (B, A)."""
    def __init__(self, temperature: float = 1.0, **_) -> None:
        self.t = temperature

    def __call__(self, pred, target):
        return F.kl_div(F.log_softmax(pred / self.t, -1), target, reduction="batchmean") * self.t**2


def default_loss_name(action: ActionSpec) -> str:
    return "cross_entropy" if action.kind == "discrete" else "mse"
