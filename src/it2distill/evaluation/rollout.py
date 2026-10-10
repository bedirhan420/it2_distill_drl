"""On-policy evaluation: the policy actually controls the (noisy) environment."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from it2distill.core.interfaces import Policy
from it2distill.envs.factory import make_env


def evaluate_returns(policy: Policy, env_id: str, sigma: float, obs_std: np.ndarray | None,
                     n_episodes: int, seed: int, noise_dims: Sequence[int] | None = None) -> np.ndarray:
    """Return an array of episode returns.

    Episode k uses env seed `seed*10_000 + k`; the noise stream is re-seeded from it on every reset
    (see GaussianObservationNoise) -> paired design across methods for each (seed, sigma, episode).
    """
    env = make_env(env_id, sigma, obs_std, seed, noise_dims)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy.act(obs))
            total, done = total + r, term or trunc
        returns.append(total)
    env.close()
    return np.asarray(returns)


def collect_obs_std(policy: Policy, env_id: str, n_episodes: int, seed: int) -> np.ndarray:
    """Per-dimension std of CLEAN observations visited by `policy`.

    Defines the scale of the relative noise (sigma_rel * std_j). Same quantity the distillation
    dataset yields, but available before WP2.
    """
    env = make_env(env_id)
    states = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        done = False
        while not done:
            states.append(obs)
            obs, _, term, trunc, _ = env.step(policy.act(obs))
            done = term or trunc
    env.close()
    return np.maximum(np.asarray(states).std(axis=0), 1e-6)


class RandomPolicy:
    """Uniform-random actions: the 0-point of the normalised-return scale."""

    def __init__(self, env_id: str, seed: int = 0) -> None:
        env = make_env(env_id)
        self.space = env.action_space
        self.space.seed(seed)
        env.close()

    def act(self, obs: np.ndarray):
        return self.space.sample()

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return np.stack([self.space.sample() for _ in range(len(obs))])


def random_baseline_return(env_id: str, n_episodes: int = 100, seed: int = 12_345) -> float:
    """Mean clean return of a uniform-random policy (R_random in the normalisation formula)."""
    policy = RandomPolicy(env_id, seed)
    return float(evaluate_returns(policy, env_id, 0.0, None, n_episodes, seed).mean())