"""T-norm Strategy: how antecedent memberships are combined into a firing strength."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from it2distill.core.registry import Registry

TNORM_REGISTRY: Registry["TNorm"] = Registry("tnorm")


class TNorm(ABC):
    @abstractmethod
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """mu: (B, R, D) -> firing strength (B, R)."""


@TNORM_REGISTRY.register("product")
class ProductTNorm(TNorm):
    def __init__(self, eps: float = 1e-9, **_) -> None:
        self.eps = eps

    def __call__(self, mu):  # log-domain: numerically stable for large D
        return torch.exp(torch.log(mu.clamp_min(self.eps)).sum(dim=-1))


@TNORM_REGISTRY.register("min")
class MinTNorm(TNorm):
    def __init__(self, **_) -> None: ...

    def __call__(self, mu):
        return mu.amin(dim=-1)
