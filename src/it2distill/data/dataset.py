"""Transition dataset + leakage-free splits (by EPISODE, not by step)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from it2distill.core.types import ActionSpec, Standardization


@dataclass
class TransitionData:
    obs: np.ndarray            # (N, D)
    actions: np.ndarray        # (N,) int or (N, A) float -- teacher labels
    episode_id: np.ndarray     # (N,)
    soft_targets: np.ndarray | None = None   # optional teacher probabilities

    def save(self, path: Path) -> None:
        np.savez_compressed(path, obs=self.obs, actions=self.actions, episode_id=self.episode_id,
                            soft=self.soft_targets if self.soft_targets is not None else np.array([]))

    @classmethod
    def load(cls, path: Path) -> "TransitionData":
        z = np.load(path)
        return cls(z["obs"], z["actions"], z["episode_id"], z["soft"] if z["soft"].size else None)

    def standardization(self) -> Standardization:
        return Standardization(self.obs.mean(0), self.obs.std(0))


@dataclass
class DataSplits:
    train: TransitionData
    val: TransitionData
    test: TransitionData


def split_by_episode(data: TransitionData, fractions: tuple[float, float, float], seed: int) -> DataSplits:
    """TODO(WP2): shuffle unique episode ids, assign to train/val/test, slice arrays accordingly."""
    raise NotImplementedError


class TorchTransitionDataset(Dataset):
    def __init__(self, data: TransitionData, action: ActionSpec) -> None:
        self.x = torch.as_tensor(data.obs, dtype=torch.float32)
        self.y = torch.as_tensor(data.actions, dtype=torch.long if action.kind == "discrete" else torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]
