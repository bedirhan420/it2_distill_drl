"""Environment Factory (+ action-space introspection)."""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np

from it2distill.core.types import ActionSpec

from .wrappers import GaussianObservationNoise


def make_env(env_id: str, sigma: float = 0.0, obs_std: np.ndarray | None = None,
             seed: int | None = None, noise_dims: Sequence[int] | None = None) -> gym.Env:
    env = gym.make(env_id)
    if sigma > 0.0:
        if obs_std is None:
            raise ValueError("obs_std required when sigma > 0")
        env = GaussianObservationNoise(env, sigma, obs_std, seed, noise_dims)
    return env


def action_spec_of(env: gym.Env) -> ActionSpec:
    sp = env.action_space
    if isinstance(sp, gym.spaces.Discrete):
        return ActionSpec("discrete", int(sp.n))
    return ActionSpec("continuous", int(np.prod(sp.shape)))