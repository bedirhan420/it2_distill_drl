"""Behaviour Strategy: WHICH policy drives the environment while the teacher labels the states.

Invariant shared by every behaviour:  stored obs = CLEAN observation,
                                      stored label = teacher action on that CLEAN observation.
Behaviours only change the state distribution that gets visited (coverage), never the label.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from it2distill.core.interfaces import Policy
from it2distill.core.registry import Registry

BEHAVIOR_REGISTRY = Registry("behavior")

SOURCE_CODES = {"clean": 0, "epsilon": 1, "noisy": 2, "dagger": 3}
SOURCE_NAMES = {v: k for k, v in SOURCE_CODES.items()}

# Fractions of the step budget given to each behaviour (A12 ablation = choice of preset).
BEHAVIOR_PRESETS = {
    "clean": {"clean": 1.0},
    "eps": {"clean": 0.5, "epsilon": 0.5},
    "noisy": {"clean": 0.5, "noisy": 0.5},
    "mixed": {"clean": 0.4, "epsilon": 0.3, "noisy": 0.3},
}


def resolve_mix(name: str) -> dict:
    if name not in BEHAVIOR_PRESETS:
        raise KeyError(f"Unknown behavior_mix '{name}'. Available: {sorted(BEHAVIOR_PRESETS)}")
    mix = BEHAVIOR_PRESETS[name]
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError(f"behavior mix '{name}' must sum to 1")
    return dict(mix)


class Behavior(ABC):
    name: str = ""

    def start_episode(self, rng: np.random.Generator) -> None:
        """Hook: per-episode randomisation (e.g. draw a noise level)."""

    @abstractmethod
    def act(self, obs: np.ndarray, label, teacher: Policy, action_space, rng: np.random.Generator):
        """Return the action actually executed. `label` = teacher action on the clean obs."""


@BEHAVIOR_REGISTRY.register("clean")
class CleanBehavior(Behavior):
    name = "clean"

    def __init__(self, **_) -> None: ...

    def act(self, obs, label, teacher, action_space, rng):
        return label


@BEHAVIOR_REGISTRY.register("epsilon")
class EpsilonBehavior(Behavior):
    """With probability epsilon take a uniformly random action (explores around the teacher path)."""
    name = "epsilon"

    def __init__(self, epsilon: float = 0.1, **_) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def act(self, obs, label, teacher, action_space, rng):
        return action_space.sample() if rng.random() < self.epsilon else label


class _ObsNoise:
    """Relative Gaussian observation noise: sigma * std_j * z, restricted to `noise_dims`."""

    def __init__(self, obs_std: np.ndarray, noise_dims: Optional[Sequence[int]]) -> None:
        std = np.asarray(obs_std, dtype=np.float64)
        mask = np.ones_like(std)
        if noise_dims is not None:
            mask = np.zeros_like(std)
            mask[list(noise_dims)] = 1.0
        self.scale = std * mask

    def apply(self, obs: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        return (obs + sigma * self.scale * rng.standard_normal(obs.shape)).astype(obs.dtype)


@BEHAVIOR_REGISTRY.register("noisy")
class NoisyBehavior(Behavior):
    """Teacher acts on a NOISY observation -> trajectories reach states a noisy deployment would visit."""
    name = "noisy"

    def __init__(self, sigmas: Sequence[float] = (1.0,), obs_std: Optional[np.ndarray] = None,
                 noise_dims: Optional[Sequence[int]] = None, **_) -> None:
        if not len(sigmas):
            raise ValueError("NoisyBehavior needs at least one sigma")
        if obs_std is None:
            raise ValueError("NoisyBehavior needs obs_std")
        self.sigmas = [float(s) for s in sigmas]
        self._noise = _ObsNoise(obs_std, noise_dims)
        self._sigma = self.sigmas[0]

    def start_episode(self, rng):
        self._sigma = float(rng.choice(self.sigmas))

    def act(self, obs, label, teacher, action_space, rng):
        return teacher.act(self._noise.apply(obs, self._sigma, rng))


@BEHAVIOR_REGISTRY.register("dagger")
class DaggerBehavior(Behavior):
    """Mixture policy for DAgger: teacher with prob `beta`, else the (optionally noise-fed) student."""
    name = "dagger"

    def __init__(self, student: Optional[Policy] = None, beta: float = 0.0, sigma: float = 0.0,
                 obs_std: Optional[np.ndarray] = None, noise_dims: Optional[Sequence[int]] = None,
                 **_) -> None:
        if student is None:
            raise ValueError("DaggerBehavior needs a student policy")
        self.student, self.beta, self.sigma = student, beta, float(sigma)
        self._noise = _ObsNoise(obs_std, noise_dims) if (sigma > 0 and obs_std is not None) else None

    def act(self, obs, label, teacher, action_space, rng):
        if rng.random() < self.beta:
            return label
        x = self._noise.apply(obs, self.sigma, rng) if self._noise is not None else obs
        return self.student.act(x)