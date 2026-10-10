"""Neural layers composing the fuzzy inference pipeline (single responsibility each)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .membership import MembershipFunction, MembershipOutput
from .tnorm import TNorm


class FuzzificationLayer(nn.Module):
    def __init__(self, mf: MembershipFunction) -> None:
        super().__init__()
        self.mf = mf

    def forward(self, x: torch.Tensor) -> MembershipOutput:
        return self.mf(x)


class RuleFiringLayer(nn.Module):
    """Applies the t-norm over antecedents; returns interval firing strengths (f_lo, f_up)."""

    def __init__(self, tnorm: TNorm) -> None:
        super().__init__()
        self.tnorm = tnorm

    def forward(self, mu: MembershipOutput) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tnorm(mu.lower), self.tnorm(mu.upper)


class TSKConsequentLayer(nn.Module):
    """Order-0: y_r = b_r.  Order-1: y_r = W_r x + b_r.   Output (B, R, A)."""

    def __init__(self, n_rules: int, n_inputs: int, n_outputs: int, order: int = 1) -> None:
        super().__init__()
        if order not in (0, 1):
            raise ValueError("order must be 0 or 1")
        self.order = order
        self.bias = nn.Parameter(torch.zeros(n_rules, n_outputs))
        self.weight = nn.Parameter(torch.zeros(n_rules, n_outputs, n_inputs)) if order == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bias.unsqueeze(0).expand(x.shape[0], -1, -1)
        if self.weight is not None:
            out = out + torch.einsum("rad,bd->bra", self.weight, x)
        return out
