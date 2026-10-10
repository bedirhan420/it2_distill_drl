"""Observer pattern: trainer emits events; callbacks handle logging / early stop / checkpoints."""
from __future__ import annotations

from abc import ABC


class TrainerCallback(ABC):
    def on_epoch_end(self, epoch: int, logs: dict) -> None: ...
    @property
    def should_stop(self) -> bool:
        return False


class EarlyStopping(TrainerCallback):
    def __init__(self, patience: int, monitor: str = "val_loss") -> None:
        self.patience, self.monitor = patience, monitor
        self.best, self.wait, self._stop = float("inf"), 0, False

    def on_epoch_end(self, epoch, logs):
        v = logs[self.monitor]
        if v < self.best - 1e-6:
            self.best, self.wait = v, 0
        else:
            self.wait += 1
            self._stop = self.wait >= self.patience

    @property
    def should_stop(self):
        return self._stop


class HistoryLogger(TrainerCallback):
    def __init__(self) -> None:
        self.history: list[dict] = []

    def on_epoch_end(self, epoch, logs):
        self.history.append({"epoch": epoch, **logs})
