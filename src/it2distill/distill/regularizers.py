"""Composable regularisers (Composite pattern). Each returns a scalar tensor."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from it2distill.fuzzy.model import FuzzyPolicyNet


class Regularizer(ABC):
    def __init__(self, weight: float) -> None:
        self.weight = weight

    @abstractmethod
    def penalty(self, model: FuzzyPolicyNet, x: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, model, x):
        return self.weight * self.penalty(model, x) if self.weight > 0 else x.new_zeros(())


class FOURegularizer(Regularizer):
    """Penalise FOU width: prevents trivially-wide FOUs (which would hide behind 'robustness')."""
    def penalty(self, model, x):
        return model.fou_width()


class RuleSparsityRegularizer(Regularizer):
    """Encourage few active rules per sample (entropy of normalised firing)  -> readability."""
    def penalty(self, model, x):
        f_lo, f_up = model.firing_strengths(x)
        p = 0.5 * (f_lo + f_up)
        p = p / (p.sum(1, keepdim=True) + 1e-9)
        return -(p * torch.log(p + 1e-9)).sum(1).mean()


class OverlapRegularizer(Regularizer):
    """TODO(WP4): penalise strongly overlapping MFs of different rules on the same input dim."""
    def penalty(self, model, x):  # pragma: no cover
        raise NotImplementedError


class CompositeRegularizer:
    def __init__(self, parts: list[Regularizer]) -> None:
        self.parts = parts

    def __call__(self, model, x):
        return sum((p(model, x) for p in self.parts), x.new_zeros(()))
