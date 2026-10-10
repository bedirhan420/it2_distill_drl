"""Statistical testing for the paper (all paired by seed)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_resamples, len(x)), replace=True).mean(1)
    return float(np.mean(x)), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def holm_correction(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def friedman_test(matrix: np.ndarray) -> float:
    """matrix: (n_blocks, n_methods). TODO(WP8): add Nemenyi / Conover post-hoc."""
    return float(stats.friedmanchisquare(*matrix.T).pvalue)
