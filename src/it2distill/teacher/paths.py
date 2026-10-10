"""Single place that defines where teacher artifacts live (DRY: scripts/pipeline share it)."""
from __future__ import annotations

from pathlib import Path

META_FILE = "teacher_meta.json"


def teacher_dir(out_dir: str | Path, name: str, seed: int) -> Path:
    return Path(out_dir) / name / f"teacher_s{seed}"


def best_model_path(directory: Path) -> Path:
    """Prefer the best checkpoint chosen by EvalCallback; fall back to the final model."""
    best = directory / "best_model.zip"
    return best if best.exists() else directory / "teacher_final.zip"


def is_trained(directory: Path) -> bool:
    """The meta file is written last, so its presence marks a COMPLETED training run."""
    return (directory / META_FILE).exists()