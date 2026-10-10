"""Ablation Command objects: each ablation = name + grid of dotted-path overrides."""
from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSpec:
    name: str
    grid: dict[str, list]

    def expand(self) -> list[dict]:
        keys = list(self.grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*self.grid.values())]


def run_tag(overrides: dict) -> str:
    return "__".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(overrides.items())) or "default"
