"""Ablation runner: expands specs, runs pipeline per (override-set, seed), appends to results CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from it2distill.config.loader import load_config
from it2distill.experiments.ablation import AblationSpec, run_tag
from it2distill.experiments.pipeline import ExperimentPipeline


def load_ablation_file(path: str | Path) -> list[AblationSpec]:
    raw = yaml.safe_load(Path(path).read_text())
    return [AblationSpec(name, spec["grid"]) for name, spec in raw["ablations"].items()]


def run_ablations(base_cfg_path: str, ablation_path: str, only: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for spec in load_ablation_file(ablation_path):
        if only and spec.name not in only:
            continue
        for overrides in spec.expand():
            cfg = load_config(base_cfg_path, overrides)
            cfg.name = f"{cfg.name}/{spec.name}/{run_tag(overrides)}"
            df = ExperimentPipeline(cfg).run_all_seeds()
            df["ablation"], df["setting"] = spec.name, run_tag(overrides)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
