"""YAML -> ExperimentConfig, with `_base_` inheritance and dotted-path overrides.

Python-version note: nested dataclasses are detected through each field's `default_factory`
instead of `typing.get_type_hints`, so annotations such as `tuple[int, ...] | None` are never
evaluated at runtime (evaluating `X | None` fails on Python 3.9).
"""
from __future__ import annotations

import copy
from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import ExperimentConfig


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_raw(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if "_base_" in raw:
        base = load_raw(path.parent / raw.pop("_base_"))
        raw = _deep_merge(base, raw)
    return raw


def apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    """overrides = {'fuzzy.n_rules': 16, 'train.lambda_fou': 0.01}"""
    raw = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return raw


def _nested_dataclass(f: Field) -> type | None:
    """Return the dataclass type of a nested config field (declared via default_factory), else None."""
    if f.default_factory is not MISSING:
        probe = f.default_factory()
        if is_dataclass(probe):
            return type(probe)
    return None


def _build(cls, data: dict):
    """Recursively instantiate dataclasses from a plain dict; unknown keys raise a clear error."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v, nested = data[f.name], _nested_dataclass(f)
        kwargs[f.name] = _build(nested, v) if nested is not None and isinstance(v, dict) else v
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw = load_raw(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(ExperimentConfig, raw)