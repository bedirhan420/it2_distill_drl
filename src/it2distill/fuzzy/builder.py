"""Builder + Factory: assembles a FuzzyPolicyNet from config; hides wiring from callers."""
from __future__ import annotations

import numpy as np
import torch

from it2distill.config.schema import FuzzyConfig
from it2distill.core.types import ActionSpec, Standardization

from .initializers import INIT_REGISTRY
from .layers import FuzzificationLayer, RuleFiringLayer, TSKConsequentLayer
from .membership import MF_REGISTRY
from .model import FuzzyPolicyNet
from .tnorm import TNORM_REGISTRY
from .type_reduction import REDUCER_REGISTRY


class FuzzyPolicyBuilder:
    """Fluent builder; `from_config` is the common entry point."""

    def __init__(self, n_inputs: int, action: ActionSpec, std: Standardization) -> None:
        self.n_inputs, self.action, self.std = n_inputs, action, std
        self._cfg: FuzzyConfig | None = None
        self._init_data: tuple[np.ndarray, np.ndarray] | None = None
        self._seed = 0

    def with_config(self, cfg: FuzzyConfig) -> "FuzzyPolicyBuilder":
        self._cfg = cfg
        return self

    def with_init_data(self, x_std: np.ndarray, y: np.ndarray, seed: int = 0) -> "FuzzyPolicyBuilder":
        self._init_data, self._seed = (x_std, y), seed
        return self

    def build(self) -> FuzzyPolicyNet:
        c = self._cfg
        if c is None:
            raise ValueError("config missing")
        mf = MF_REGISTRY.create(c.membership, n_rules=c.n_rules, n_inputs=self.n_inputs,
                                fou_init=c.fou_init, fou_learnable=c.fou_learnable)
        if self._init_data is not None:
            centers, sigmas = INIT_REGISTRY.create(c.initializer)(*self._init_data, c.n_rules, self._seed)
            mf.initialize(torch.as_tensor(centers, dtype=torch.float32),
                          torch.as_tensor(sigmas, dtype=torch.float32))
        return FuzzyPolicyNet(
            fuzzify=FuzzificationLayer(mf),
            firing=RuleFiringLayer(TNORM_REGISTRY.create(c.tnorm)),
            consequent=TSKConsequentLayer(c.n_rules, self.n_inputs, self.action.dim, c.consequent_order),
            reducer=REDUCER_REGISTRY.create(c.reducer),
            standardization=self.std,
        )

    @classmethod
    def from_config(cls, cfg: FuzzyConfig, n_inputs: int, action: ActionSpec, std: Standardization,
                    x_train: np.ndarray, y_train: np.ndarray, seed: int = 0) -> FuzzyPolicyNet:
        x_std = (x_train - std.mean) / np.maximum(std.std, 1e-6)
        return cls(n_inputs, action, std).with_config(cfg).with_init_data(x_std, y_train, seed).build()
