"""Membership functions. Every MF returns a (lower, upper) pair.

T1 MFs return lower == upper, so downstream code is type-agnostic (Liskov).
Shapes: x (B, D) -> (B, R, D) where R = rules, D = inputs. Rule-wise MFs mean the rule
count is chosen directly (no K^D grid explosion) and each rule stays individually readable.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from it2distill.core.registry import Registry

MF_REGISTRY: Registry["MembershipFunction"] = Registry("membership")


@dataclass
class MembershipOutput:
    lower: torch.Tensor
    upper: torch.Tensor


class MembershipFunction(nn.Module, ABC):
    def __init__(self, n_rules: int, n_inputs: int) -> None:
        super().__init__()
        self.n_rules, self.n_inputs = n_rules, n_inputs

    @property
    @abstractmethod
    def is_interval(self) -> bool: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> MembershipOutput: ...

    @abstractmethod
    def center_and_width(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(R, D) centers and nominal sigmas, used for rule extraction / linguistic labels."""

    def fou_width(self) -> torch.Tensor:
        """Mean relative FOU width (0 for T1). Used by the FOU regulariser."""
        return torch.zeros((), device=self.center_and_width()[0].device)

    def initialize(self, centers: torch.Tensor, sigmas: torch.Tensor) -> None:
        with torch.no_grad():
            self.centers.copy_(centers)
            self.log_sigma.copy_(sigmas.clamp_min(1e-3).log())


def _gauss(x, c, s):
    return torch.exp(-0.5 * ((x - c) / s) ** 2)


@MF_REGISTRY.register("gaussian")
class GaussianMF(MembershipFunction):
    """Type-1 Gaussian."""

    def __init__(self, n_rules: int, n_inputs: int, **_) -> None:
        super().__init__(n_rules, n_inputs)
        self.centers = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.log_sigma = nn.Parameter(torch.zeros(n_rules, n_inputs))

    is_interval = property(lambda self: False)

    def forward(self, x):
        mu = _gauss(x[:, None, :], self.centers[None], self.log_sigma.exp()[None])
        return MembershipOutput(mu, mu)

    def center_and_width(self):
        return self.centers.detach(), self.log_sigma.exp().detach()


@MF_REGISTRY.register("it2_uncertain_sigma")
class IT2GaussianUncertainSigma(MembershipFunction):
    """Fixed mean, sigma in [sigma, sigma*(1+softplus(rho))]  ->  lower <= upper by construction."""

    def __init__(self, n_rules: int, n_inputs: int, fou_init: float = 0.3,
                 fou_learnable: bool = True, **_) -> None:
        super().__init__(n_rules, n_inputs)
        self.centers = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.log_sigma = nn.Parameter(torch.zeros(n_rules, n_inputs))
        # inverse softplus so that softplus(rho) == fou_init at start
        rho0 = math.log(math.expm1(fou_init))
        self.rho = nn.Parameter(torch.full((n_rules, n_inputs), rho0), requires_grad=fou_learnable)

    is_interval = property(lambda self: True)

    def _ratio(self):
        return F.softplus(self.rho)

    def forward(self, x):
        s_lo = self.log_sigma.exp()[None]
        s_up = s_lo * (1.0 + self._ratio()[None])
        xx, c = x[:, None, :], self.centers[None]
        return MembershipOutput(lower=_gauss(xx, c, s_lo), upper=_gauss(xx, c, s_up))

    def center_and_width(self):
        return self.centers.detach(), self.log_sigma.exp().detach()

    def fou_width(self):
        return self._ratio().mean()


@MF_REGISTRY.register("it2_uncertain_mean")
class IT2GaussianUncertainMean(MembershipFunction):
    """Fixed sigma, mean in [c-h, c+h]. Upper MF has a flat top of width 2h."""

    def __init__(self, n_rules: int, n_inputs: int, fou_init: float = 0.3,
                 fou_learnable: bool = True, **_) -> None:
        super().__init__(n_rules, n_inputs)
        self.centers = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.log_sigma = nn.Parameter(torch.zeros(n_rules, n_inputs))
        rho0 = math.log(math.expm1(fou_init))
        self.rho = nn.Parameter(torch.full((n_rules, n_inputs), rho0), requires_grad=fou_learnable)

    is_interval = property(lambda self: True)

    def _half_width(self):  # h = sigma * softplus(rho): FOU scale-free w.r.t. input units
        return self.log_sigma.exp() * F.softplus(self.rho)

    def forward(self, x):
        xx, c, s, h = x[:, None, :], self.centers[None], self.log_sigma.exp()[None], self._half_width()[None]
        d = (xx - c).abs()
        upper = torch.exp(-0.5 * (F.relu(d - h) / s) ** 2)
        lower = torch.exp(-0.5 * ((d + h) / s) ** 2)
        return MembershipOutput(lower, upper)

    def center_and_width(self):
        return self.centers.detach(), self.log_sigma.exp().detach()

    def fou_width(self):
        return F.softplus(self.rho).mean()
