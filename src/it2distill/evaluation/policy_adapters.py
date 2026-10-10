"""Adapters turning students into the `Policy` protocol (torch / sklearn)."""
from __future__ import annotations

import numpy as np
import torch

from it2distill.core.types import ActionSpec


class TorchPolicy:
    def __init__(self, model: torch.nn.Module, action: ActionSpec) -> None:
        self.model, self.action = model.eval(), action

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        out = self.model(torch.as_tensor(obs, dtype=torch.float32))
        return out.argmax(-1).numpy() if self.action.kind == "discrete" else out.numpy()

    def act(self, obs: np.ndarray):
        a = self.act_batch(obs[None])[0]
        return int(a) if self.action.kind == "discrete" else a
