"""Shared lightweight value objects (no heavy dependencies)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActionKind = Literal["discrete", "continuous"]


@dataclass(frozen=True)
class ActionSpec:
    """Describes the action space; drives loss / head selection (no if-chains elsewhere)."""
    kind: ActionKind
    dim: int  # n_actions (discrete) or action vector size (continuous)


@dataclass(frozen=True)
class Standardization:
    """Per-feature statistics estimated on the TRAIN split only (avoid leakage)."""
    mean: np.ndarray
    std: np.ndarray
