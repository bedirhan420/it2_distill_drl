"""Persistence of splits + metadata. `data_meta.json` is written LAST = completion marker."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .dataset import DataSplits, TransitionData

META_FILE = "data_meta.json"
_SPLITS = ("train", "val", "test")


def data_dir(teacher_directory: Path) -> Path:
    return Path(teacher_directory) / "data"


def is_collected(directory: Path) -> bool:
    return (Path(directory) / META_FILE).exists()


def _save_one(data: TransitionData, path: Path) -> None:
    soft = data.soft_targets if data.soft_targets is not None else np.zeros((0,), dtype=np.float32)
    np.savez_compressed(path, obs=data.obs, actions=data.actions, episode_id=data.episode_id,
                        source=data.source, soft=soft)


def _load_one(path: Path) -> TransitionData:
    z = np.load(path, allow_pickle=False)
    return TransitionData(z["obs"], z["actions"], z["episode_id"], z["source"],
                          z["soft"] if z["soft"].size else None)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_splits(splits: DataSplits, directory: Path, meta: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in _SPLITS:
        _save_one(getattr(splits, name), directory / f"{name}.npz")
    (directory / META_FILE).write_text(json.dumps(meta, indent=2, default=_jsonable))


def load_splits(directory: Path) -> Tuple[DataSplits, dict]:
    directory = Path(directory)
    if not is_collected(directory):
        raise FileNotFoundError(f"no completed dataset in {directory} (run `make data`)")
    splits = DataSplits(*[_load_one(directory / f"{n}.npz") for n in _SPLITS])
    return splits, json.loads((directory / META_FILE).read_text())