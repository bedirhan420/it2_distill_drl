"""Distillation trainer: model/loss/regulariser/augmenter injected (Dependency Injection)."""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch.utils.data import DataLoader

from it2distill.config.schema import TrainConfig

from .callbacks import TrainerCallback
from .losses import DistillLoss


class DistillationTrainer:
    def __init__(self, model: torch.nn.Module, loss: DistillLoss, cfg: TrainConfig,
                 regularizer: Callable | None = None, augment: Callable | None = None,
                 callbacks: list[TrainerCallback] | None = None, device: str = "cpu") -> None:
        self.model, self.loss, self.cfg = model.to(device), loss, cfg
        self.reg, self.augment, self.callbacks, self.device = regularizer, augment, callbacks or [], device
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _step(self, x, y, train: bool) -> float:
        x, y = x.to(self.device), y.to(self.device)
        if train and self.augment is not None:
            x = self.augment(x)
        pred = self.model(x)
        loss = self.loss(pred, y)
        if train and self.reg is not None:
            loss = loss + self.reg(self.model, x)
        if train:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return float(loss.detach())

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> torch.nn.Module:
        best_state, best_val = None, float("inf")
        for epoch in range(self.cfg.epochs):
            self.model.train()
            tr = sum(self._step(x, y, True) for x, y in train_loader) / len(train_loader)
            self.model.eval()
            with torch.no_grad():
                va = sum(self._step(x, y, False) for x, y in val_loader) / len(val_loader)
            if va < best_val:
                best_val, best_state = va, copy.deepcopy(self.model.state_dict())
            logs = {"train_loss": tr, "val_loss": va}
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs)
            if any(cb.should_stop for cb in self.callbacks):
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.model
