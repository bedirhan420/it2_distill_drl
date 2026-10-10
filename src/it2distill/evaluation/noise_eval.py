"""NoiseRobustnessEvaluator: runs {policy} x {sigma} x {seed} and returns a tidy DataFrame."""
from __future__ import annotations

import pandas as pd

from it2distill.config.schema import EvalConfig
from it2distill.core.interfaces import Policy

from .rollout import evaluate_returns


class NoiseRobustnessEvaluator:
    def __init__(self, env_id: str, obs_std, cfg: EvalConfig) -> None:
        self.env_id, self.obs_std, self.cfg = env_id, obs_std, cfg

    def run(self, policies: dict[str, Policy]) -> pd.DataFrame:
        """TODO(WP7): for name,pol in policies; for sigma; for seed -> rows
        (method, sigma, seed, mean_return, fidelity_vs_clean_teacher). Cache to parquet."""
        raise NotImplementedError
