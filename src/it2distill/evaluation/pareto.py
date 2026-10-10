"""Pareto-front utilities: (n_rules, objective) trade-offs, e.g. rules vs robustness AUC."""
from __future__ import annotations

import numpy as np


def pareto_front(points: np.ndarray, minimize: tuple[bool, ...]) -> np.ndarray:
    """Return boolean mask of non-dominated rows. points: (N, M)."""
    p = points * np.where(minimize, 1.0, -1.0)
    mask = np.ones(len(p), dtype=bool)
    for i in range(len(p)):
        dominated = np.all(p <= p[i], axis=1) & np.any(p < p[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def hypervolume_2d(front: np.ndarray, ref: tuple[float, float]) -> float:
    """TODO(WP8): scalar front-quality indicator to compare methods' Pareto fronts."""
    raise NotImplementedError
