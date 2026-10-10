"""Rollout collection from the teacher. Supports epsilon-perturbed behaviour (state coverage)
and DAgger rounds (student drives, teacher labels) to fight covariate shift."""
from __future__ import annotations

import numpy as np

from it2distill.config.schema import DataConfig
from it2distill.core.interfaces import Policy
from it2distill.envs.factory import make_env

from .dataset import TransitionData


class TeacherDataCollector:
    def __init__(self, env_id: str, teacher: Policy, cfg: DataConfig, seed: int) -> None:
        self.env_id, self.teacher, self.cfg, self.seed = env_id, teacher, cfg, seed

    def collect(self) -> TransitionData:
        """TODO(WP2): loop episodes until cfg.n_steps; with prob collection_epsilon take a random
        action to step the env but ALWAYS record the teacher's label for the visited state."""
        raise NotImplementedError

    def collect_dagger(self, student: Policy, n_steps: int, beta: float) -> TransitionData:
        """TODO(WP2/A9): mixture policy (beta*teacher + (1-beta)*student) drives, teacher labels."""
        raise NotImplementedError
