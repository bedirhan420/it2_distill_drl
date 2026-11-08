"""Teacher rollout collection (behavioural cloning data + DAgger rounds).

Seeds: every episode uses env seed  EP_SEED_BASE + teacher_seed*TEACHER_STRIDE + k.
That range is disjoint from the evaluation seeds used in WP1/WP7 (< 1_200_000), so no evaluation
initial state is ever seen in training data.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from it2distill.config.schema import DataConfig
from it2distill.core.interfaces import Policy
from it2distill.envs.factory import action_spec_of, make_env
from it2distill.evaluation.rollout import collect_obs_std

from .behaviors import (BEHAVIOR_REGISTRY, SOURCE_CODES, Behavior, DaggerBehavior, resolve_mix)
from .dataset import TransitionData

EP_SEED_BASE = 5_000_000
TEACHER_STRIDE = 100_000
DAGGER_OFFSET = 50_000
DAGGER_ROUND_STRIDE = 10_000
MAX_DAGGER_ROUNDS = 5
OBS_STD_SEED = 999            # same as scripts/07_eval_teacher.py -> identical noise scale everywhere
_SOFT_CHUNK = 10_000


class _Recorder:
    def __init__(self) -> None:
        self.obs, self.act, self.ep, self.src = [], [], [], []

    def add(self, obs, label, episode_id: int, source_code: int) -> None:
        self.obs.append(obs)
        self.act.append(label)
        self.ep.append(episode_id)
        self.src.append(source_code)

    def to_data(self, discrete: bool) -> TransitionData:
        actions = np.asarray(self.act, dtype=np.int64 if discrete else np.float32)
        return TransitionData(np.asarray(self.obs, dtype=np.float32), actions,
                              np.asarray(self.ep, dtype=np.int64), np.asarray(self.src, dtype=np.int8))


class TeacherDataCollector:
    def __init__(self, env_id: str, teacher: Policy, cfg: DataConfig, seed: int,
                 noise_dims: Optional[Sequence[int]] = None, obs_std: Optional[np.ndarray] = None) -> None:
        self.env_id, self.teacher, self.cfg, self.seed = env_id, teacher, cfg, seed
        self.noise_dims = noise_dims
        self._obs_std = None if obs_std is None else np.asarray(obs_std, dtype=np.float64)

    # ------------------------------------------------------------------ helpers
    @property
    def obs_std(self) -> np.ndarray:
        """Std of CLEAN teacher states (scale of the relative noise); estimated once and cached."""
        if self._obs_std is None:
            self._obs_std = collect_obs_std(self.teacher, self.env_id, self.cfg.obs_std_episodes,
                                            OBS_STD_SEED)
        return self._obs_std

    def _base_seed(self) -> int:
        return EP_SEED_BASE + self.seed * TEACHER_STRIDE

    def _run_episode(self, env, behavior: Behavior, source_code: int, ep_seed: int,
                     episode_id: int, rec: _Recorder) -> int:
        rng = np.random.default_rng(ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        env.action_space.seed(ep_seed)
        behavior.start_episode(rng)
        steps, done = 0, False
        while not done:
            label = self.teacher.act(obs)                       # label on the CLEAN observation
            action = behavior.act(obs, label, self.teacher, env.action_space, rng)
            rec.add(obs, label, episode_id, source_code)
            obs, _, term, trunc, _ = env.step(action)
            done, steps = term or trunc, steps + 1
        return steps

    def _attach_soft_targets(self, data: TransitionData) -> TransitionData:
        if not self.cfg.save_soft_targets or not hasattr(self.teacher, "soft_targets"):
            return data
        if data.actions.ndim != 1:        # discrete actions only
            return data
        chunks = [self.teacher.soft_targets(data.obs[i:i + _SOFT_CHUNK])
                  for i in range(0, len(data), _SOFT_CHUNK)]
        data.soft_targets = np.concatenate(chunks).astype(np.float32)
        return data

    # ------------------------------------------------------------------ collection
    def collect(self) -> TransitionData:
        """Behavioural-cloning dataset following the configured behaviour mix."""
        mix = resolve_mix(self.cfg.behavior_mix)
        env = make_env(self.env_id)
        discrete = action_spec_of(env).kind == "discrete"
        behaviors = {name: BEHAVIOR_REGISTRY.create(
            name, epsilon=self.cfg.collection_epsilon, sigmas=self.cfg.behavior_noise_sigmas,
            obs_std=self.obs_std if name == "noisy" else None, noise_dims=self.noise_dims)
            for name in mix}

        rec, ep = _Recorder(), 0
        for name, frac in mix.items():
            quota, steps = int(round(frac * self.cfg.n_steps)), 0
            while steps < quota:     # whole episodes only -> a source may overshoot its quota slightly
                if ep >= DAGGER_OFFSET:
                    raise RuntimeError("too many episodes; would collide with DAgger seed range")
                steps += self._run_episode(env, behaviors[name], SOURCE_CODES[name],
                                           self._base_seed() + ep, ep, rec)
                ep += 1
        env.close()
        return self._attach_soft_targets(rec.to_data(discrete))

    def collect_dagger(self, student: Policy, n_steps: int, beta: float, round_idx: int,
                       noise_sigma: float = 0.0) -> TransitionData:
        """One DAgger round: the mixture policy drives, the teacher labels every visited (clean) state.

        round_idx in [0, MAX_DAGGER_ROUNDS) selects a disjoint env-seed block. If noise_sigma > 0 the
        student is fed noisy observations, so the dataset covers states a noisy deployment would reach.
        """
        if not 0 <= round_idx < MAX_DAGGER_ROUNDS:
            raise ValueError(f"round_idx must be in [0, {MAX_DAGGER_ROUNDS})")
        env = make_env(self.env_id)
        discrete = action_spec_of(env).kind == "discrete"
        behavior = DaggerBehavior(student=student, beta=beta, sigma=noise_sigma,
                                  obs_std=self.obs_std if noise_sigma > 0 else None,
                                  noise_dims=self.noise_dims)
        base = self._base_seed() + DAGGER_OFFSET + round_idx * DAGGER_ROUND_STRIDE
        rec, ep, steps = _Recorder(), 0, 0
        while steps < n_steps:
            steps += self._run_episode(env, behavior, SOURCE_CODES["dagger"], base + ep, ep, rec)
            ep += 1
        env.close()
        return self._attach_soft_targets(rec.to_data(discrete))