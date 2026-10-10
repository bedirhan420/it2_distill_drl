"""Observation-noise Decorator (gymnasium wrapper). Noise is RELATIVE to per-dim obs std so that
sigma levels are comparable across environments and feature scales.

* `noise_dims` restricts noise to selected observation dims (e.g. skip binary leg-contact flags).
* Pairing guarantee: on `reset(seed=s)` the noise RNG is re-seeded from `s`, so the k-th noisy
  observation of an episode is IDENTICAL for every policy evaluated with the same (seed, episode),
  regardless of how long each policy keeps the episode alive. This makes paired tests valid.
"""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np


class GaussianObservationNoise(gym.ObservationWrapper):
    _NOISE_SEED_OFFSET = 12_345

    def __init__(self, env: gym.Env, sigma: float, obs_std: np.ndarray, seed: int | None = None,
                 noise_dims: Sequence[int] | None = None) -> None:
        super().__init__(env)
        self.sigma = float(sigma)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        if self.obs_std.ndim != 1:
            raise ValueError("obs_std must be 1-D (flat observation vector)")
        mask = np.ones_like(self.obs_std)
        if noise_dims is not None:
            if any(not 0 <= d < mask.size for d in noise_dims):
                raise ValueError(f"noise_dims {list(noise_dims)} out of range for obs size {mask.size}")
            mask = np.zeros_like(self.obs_std)
            mask[list(noise_dims)] = 1.0
        self._scale = (self.sigma * self.obs_std * mask).astype(np.float32)  # per-dim noise std
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + self._NOISE_SEED_OFFSET)
        return super().reset(seed=seed, options=options)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.sigma == 0.0:
            return obs
        return (obs + self._rng.standard_normal(obs.shape) * self._scale).astype(np.float32)