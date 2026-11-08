"""Torch adapters (kept separate so the data core stays numpy-only and easy to test)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from it2distill.core.types import ActionSpec

from .dataset import TransitionData


class TorchTransitionDataset(Dataset):
    def __init__(self, data: TransitionData, action: ActionSpec, use_soft_targets: bool = False) -> None:
        self.x = torch.as_tensor(data.obs, dtype=torch.float32)
        if use_soft_targets:
            if data.soft_targets is None:
                raise ValueError("dataset has no soft targets (re-collect with save_soft_targets=true)")
            self.y = torch.as_tensor(data.soft_targets, dtype=torch.float32)
        else:
            self.y = torch.as_tensor(data.actions,
                                     dtype=torch.long if action.kind == "discrete" else torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def make_loader(data: TransitionData, action: ActionSpec, batch_size: int, shuffle: bool,
                seed: int = 0, use_soft_targets: bool = False) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(TorchTransitionDataset(data, action, use_soft_targets), batch_size=batch_size,
                      shuffle=shuffle, generator=gen if shuffle else None)