"""Abstractions (Dependency Inversion): high-level code depends on these, not on concretes."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Anything that maps an observation to an action: teacher, fuzzy student, tree, MLP."""

    def act(self, obs: np.ndarray) -> np.ndarray | int: ...
    def act_batch(self, obs: np.ndarray) -> np.ndarray: ...


class Explainable(ABC):
    """Policies that can expose a human-readable rule base."""

    @abstractmethod
    def n_rules(self) -> int: ...

    @abstractmethod
    def extract_rules(self) -> Sequence["Rule"]: ...


class Rule(Protocol):
    antecedent: Sequence[str]
    consequent: str
    firing_note: str
