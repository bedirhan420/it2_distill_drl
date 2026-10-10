"""Noise-augmentation Strategy for training inputs (ablation A7: clean vs noise-aware distillation)."""
from __future__ import annotations

import torch


class GaussianAugment:
    def __init__(self, sigma: float, obs_std: torch.Tensor) -> None:
        self.sigma, self.obs_std = sigma, obs_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return x
        return x + torch.randn_like(x) * self.sigma * self.obs_std.to(x.device)
