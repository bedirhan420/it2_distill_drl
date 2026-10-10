"""Rule-base initialisation Strategy: data-driven starts make rule counts small and training stable."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from it2distill.core.registry import Registry

INIT_REGISTRY: Registry["RuleInitializer"] = Registry("initializer")


class RuleInitializer(ABC):
    @abstractmethod
    def __call__(self, x_std: np.ndarray, y: np.ndarray, n_rules: int, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Return (centers (R,D), sigmas (R,D)) in standardised input space."""


@INIT_REGISTRY.register("random")
class RandomInit(RuleInitializer):
    def __init__(self, **_) -> None: ...

    def __call__(self, x_std, y, n_rules, seed):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_std), n_rules, replace=False)
        return x_std[idx], np.ones((n_rules, x_std.shape[1]))


@INIT_REGISTRY.register("kmeans")
class KMeansInit(RuleInitializer):
    """Cluster in (state, action) space so rules specialise on action-relevant regions."""

    def __init__(self, action_weight: float = 1.0, **_) -> None:
        self.action_weight = action_weight

    def __call__(self, x_std, y, n_rules, seed):
        from sklearn.cluster import KMeans
        y2 = y.reshape(len(y), -1).astype(float)
        z = np.concatenate([x_std, self.action_weight * y2], axis=1)
        km = KMeans(n_clusters=n_rules, n_init=10, random_state=seed).fit(z)
        d = x_std.shape[1]
        centers = km.cluster_centers_[:, :d]
        sigmas = np.stack([x_std[km.labels_ == k].std(0) + 1e-2 for k in range(n_rules)])
        return centers, sigmas


@INIT_REGISTRY.register("teacher_tree")
class TeacherTreeInit(RuleInitializer):
    """TODO(WP4 ablation A4): fit a shallow tree on (x, teacher action), use leaf boxes as rule seeds."""

    def __init__(self, **_) -> None: ...

    def __call__(self, x_std, y, n_rules, seed):  # pragma: no cover
        raise NotImplementedError
