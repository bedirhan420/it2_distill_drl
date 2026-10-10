"""Competing students sharing the same distillation data & evaluation harness."""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text


class DecisionTreeStudent:
    """Interpretable baseline (cf. VIPER / Bastani et al.). Rule count = number of leaves."""

    def __init__(self, discrete: bool, max_leaf_nodes: int, seed: int = 0) -> None:
        cls = DecisionTreeClassifier if discrete else DecisionTreeRegressor
        self.tree = cls(max_leaf_nodes=max_leaf_nodes, random_state=seed)
        self.discrete = discrete

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeStudent":
        self.tree.fit(x, y)
        return self

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return self.tree.predict(obs)

    def act(self, obs: np.ndarray):
        a = self.act_batch(obs[None])[0]
        return int(a) if self.discrete else a

    def n_rules(self) -> int:
        return int(self.tree.get_n_leaves())

    def as_text(self, names) -> str:
        return export_text(self.tree, feature_names=list(names))


def make_mlp_student(n_inputs: int, out_dim: int, hidden: int, depth: int = 2):
    """Parameter-matched black-box student (upper bound on fidelity; shows cost of interpretability)."""
    import torch.nn as nn
    layers, d = [], n_inputs
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    return nn.Sequential(*layers, nn.Linear(d, out_dim))
