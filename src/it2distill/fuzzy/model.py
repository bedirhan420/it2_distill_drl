"""FuzzyPolicyNet: T1 and IT2 are the SAME class with different injected components (Composition)."""
from __future__ import annotations

import torch
import torch.nn as nn

from it2distill.core.interfaces import Explainable
from it2distill.core.types import Standardization

from .layers import FuzzificationLayer, RuleFiringLayer, TSKConsequentLayer
from .type_reduction import Reducer


class FuzzyPolicyNet(nn.Module, Explainable):
    def __init__(self, fuzzify: FuzzificationLayer, firing: RuleFiringLayer,
                 consequent: TSKConsequentLayer, reducer: Reducer,
                 standardization: Standardization) -> None:
        super().__init__()
        self.fuzzify, self.firing, self.consequent, self.reducer = fuzzify, firing, consequent, reducer
        self.register_buffer("x_mean", torch.as_tensor(standardization.mean, dtype=torch.float32))
        self.register_buffer("x_std", torch.as_tensor(standardization.std, dtype=torch.float32).clamp_min(1e-6))

    # ---- inference ---------------------------------------------------
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.x_mean) / self.x_std
        f_lo, f_up = self.firing(self.fuzzify(x))
        return self.reducer(f_lo, f_up, self.consequent(x))

    def firing_strengths(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (obs - self.x_mean) / self.x_std
        return self.firing(self.fuzzify(x))

    # ---- regularisation hooks (used by distill.regularizers) -------------
    def fou_width(self) -> torch.Tensor:
        return self.fuzzify.mf.fou_width()

    # ---- Explainable ---------------------------------------------------
    def n_rules(self) -> int:
        return self.fuzzify.mf.n_rules

    def extract_rules(self):
        from it2distill.explain.rule_extractor import RuleExtractor
        return RuleExtractor().extract(self)
