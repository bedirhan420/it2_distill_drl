"""Adapter: SB3 model -> our `Policy` protocol (so the evaluator never imports SB3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


class SB3TeacherPolicy:
    def __init__(self, path: str | Path, deterministic: bool = True) -> None:
        self.model = PPO.load(str(path), device="cpu")
        self.deterministic = deterministic

    def act(self, obs: np.ndarray):
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a) if np.ndim(a) == 0 else a  # plain int for Discrete spaces

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return a

    def soft_targets(self, obs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Teacher action probabilities (discrete) for KL / soft-label distillation (ablation A8)."""
        import torch
        obs_t, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            logits = self.model.policy.get_distribution(obs_t).distribution.logits
        return torch.softmax(logits / temperature, dim=-1).cpu().numpy()