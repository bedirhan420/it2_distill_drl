"""Type-reduction / defuzzification Strategy.

Reducers consume (f_lo, f_up: (B,R)) and rule outputs y: (B,R,A) and return (B,A).
Karnik-Mendel is iterative and non-differentiable -> `KarnikMendelReference` is for EVALUATION ONLY,
used to quantify the approximation gap of the differentiable reducers (ablation A5).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from it2distill.core.registry import Registry

REDUCER_REGISTRY: Registry["Reducer"] = Registry("reducer")
_EPS = 1e-9


class Reducer(nn.Module, ABC):
    @abstractmethod
    def forward(self, f_lo: torch.Tensor, f_up: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


@REDUCER_REGISTRY.register("weighted_average")
class WeightedAverage(Reducer):
    """T1 defuzzification (uses lower==upper; takes the mean for safety)."""

    def __init__(self, **_) -> None:
        super().__init__()

    def forward(self, f_lo, f_up, y):
        f = 0.5 * (f_lo + f_up)
        return (f.unsqueeze(-1) * y).sum(1) / (f.sum(1, keepdim=True) + _EPS)


@REDUCER_REGISTRY.register("nie_tan")
class NieTan(Reducer):
    """Closed-form, differentiable: weights = midpoint of the firing interval."""

    def __init__(self, **_) -> None:
        super().__init__()

    def forward(self, f_lo, f_up, y):
        f = 0.5 * (f_lo + f_up)
        return (f.unsqueeze(-1) * y).sum(1) / (f.sum(1, keepdim=True) + _EPS)


@REDUCER_REGISTRY.register("bounds_mixing")
class BoundsMixing(Reducer):
    """Learnable convex mix of lower- and upper-firing weighted averages (BM-style approximation)."""

    def __init__(self, **_) -> None:
        super().__init__()
        self.q_logit = nn.Parameter(torch.zeros(1))

    def forward(self, f_lo, f_up, y):
        y_lo = (f_lo.unsqueeze(-1) * y).sum(1) / (f_lo.sum(1, keepdim=True) + _EPS)
        y_up = (f_up.unsqueeze(-1) * y).sum(1) / (f_up.sum(1, keepdim=True) + _EPS)
        q = torch.sigmoid(self.q_logit)
        return q * y_lo + (1 - q) * y_up


class KarnikMendelReference:
    """TODO(WP5): exact KM algorithm (numpy, per-sample, per-output) for evaluation-only comparison."""

    def __call__(self, f_lo, f_up, y):  # pragma: no cover
        raise NotImplementedError
