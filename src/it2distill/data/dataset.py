"""Transition container + leakage-free, source-stratified splits by EPISODE (numpy only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from it2distill.core.types import Standardization


@dataclass
class TransitionData:
    obs: np.ndarray                       # (N, D) float32 -- CLEAN observations
    actions: np.ndarray                   # (N,) int64 or (N, A) float32 -- teacher labels on CLEAN obs
    episode_id: np.ndarray                # (N,) int64
    source: np.ndarray                    # (N,) int8 -- behaviour code (see behaviors.SOURCE_CODES)
    soft_targets: Optional[np.ndarray] = None   # (N, A) teacher probabilities (discrete only)

    def __post_init__(self) -> None:
        n = len(self.obs)
        parts = {"actions": self.actions, "episode_id": self.episode_id, "source": self.source}
        if self.soft_targets is not None:
            parts["soft_targets"] = self.soft_targets
        for name, arr in parts.items():
            if len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)} but obs has {n}")

    def __len__(self) -> int:
        return len(self.obs)

    @property
    def n_episodes(self) -> int:
        return int(len(np.unique(self.episode_id)))

    def subset(self, mask: np.ndarray) -> "TransitionData":
        return TransitionData(self.obs[mask], self.actions[mask], self.episode_id[mask],
                              self.source[mask],
                              None if self.soft_targets is None else self.soft_targets[mask])

    def standardization(self) -> Standardization:
        return Standardization(self.obs.mean(0), self.obs.std(0))

    @staticmethod
    def concat(parts: Sequence["TransitionData"]) -> "TransitionData":
        """Aggregate datasets (e.g. DAgger rounds). Episode ids are offset so they stay unique."""
        offset, ids = 0, []
        for p in parts:
            ids.append(p.episode_id + offset)
            offset += int(p.episode_id.max()) + 1 if len(p) else 0
        soft = None
        if all(p.soft_targets is not None for p in parts):
            soft = np.concatenate([p.soft_targets for p in parts])
        return TransitionData(np.concatenate([p.obs for p in parts]),
                              np.concatenate([p.actions for p in parts]),
                              np.concatenate(ids), np.concatenate([p.source for p in parts]), soft)


@dataclass
class DataSplits:
    train: TransitionData
    val: TransitionData
    test: TransitionData

    def sizes(self) -> dict:
        return {k: len(getattr(self, k)) for k in ("train", "val", "test")}


def split_by_episode(data: TransitionData, fractions: Sequence[float], seed: int) -> DataSplits:
    """Assign WHOLE episodes to train/val/test, stratified by behaviour source.

    Steps within one episode are strongly correlated; a step-level split would leak near-duplicate
    states into val/test and inflate fidelity. `fractions` are fractions of EPISODES per source,
    so the step fractions are only approximate.
    """
    f_train, f_val, f_test = fractions
    if abs(f_train + f_val + f_test - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    train_eps, val_eps, test_eps = [], [], []
    for code in np.unique(data.source):
        eps = np.unique(data.episode_id[data.source == code])
        if len(eps) < 3:
            raise ValueError(f"source {int(code)} has only {len(eps)} episode(s); need >= 3 to split. "
                             "Collect more steps or use a smaller behaviour share.")
        eps = rng.permutation(eps)
        n_val = max(1, int(round(f_val * len(eps))))
        n_test = max(1, int(round(f_test * len(eps))))
        if len(eps) - n_val - n_test < 1:
            raise ValueError("not enough episodes left for the train split")
        val_eps.append(eps[:n_val])
        test_eps.append(eps[n_val:n_val + n_test])
        train_eps.append(eps[n_val + n_test:])

    def take(ep_lists):
        return data.subset(np.isin(data.episode_id, np.concatenate(ep_lists)))

    splits = DataSplits(take(train_eps), take(val_eps), take(test_eps))
    ids = [set(np.unique(s.episode_id).tolist()) for s in (splits.train, splits.val, splits.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "episode leakage between splits"
    return splits