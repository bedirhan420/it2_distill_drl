"""Dataset diagnostics: verifies that the behaviour mix really widens state coverage."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .behaviors import SOURCE_NAMES
from .dataset import TransitionData


def summarize(data: TransitionData) -> dict:
    """Per-source step/episode counts, mean episode length and state-spread ratio vs the clean source
    (mean over dims of std_source / std_clean; > 1 means the source visits a wider region)."""
    out = {"n_steps": len(data), "n_episodes": data.n_episodes, "per_source": {}}
    clean_std: Optional[np.ndarray] = None
    if 0 in np.unique(data.source):
        clean_std = data.obs[data.source == 0].std(0)
    for code in np.unique(data.source):
        m = data.source == code
        eps = data.episode_id[m]
        entry = {"steps": int(m.sum()), "episodes": int(len(np.unique(eps))),
                 "mean_episode_len": float(m.sum() / len(np.unique(eps)))}
        if clean_std is not None:
            valid = clean_std > 1e-6
            entry["state_spread_vs_clean"] = float(np.mean(data.obs[m].std(0)[valid] / clean_std[valid]))
        out["per_source"][SOURCE_NAMES[int(code)]] = entry
    if data.actions.ndim == 1:
        vals, counts = np.unique(data.actions, return_counts=True)
        out["action_distribution"] = {int(v): float(c / len(data)) for v, c in zip(vals, counts)}
    return out


def render(summary: dict) -> str:
    lines = [f"  total: {summary['n_steps']:,} steps, {summary['n_episodes']} episodes"]
    for name, e in summary["per_source"].items():
        spread = e.get("state_spread_vs_clean")
        sp = f", state spread x{spread:.2f} vs clean" if spread is not None else ""
        lines.append(f"  {name:<8} {e['steps']:>7,} steps | {e['episodes']:>4} ep | "
                     f"mean len {e['mean_episode_len']:6.1f}{sp}")
    if "action_distribution" in summary:
        lines.append("  action distribution: " + ", ".join(
            f"{k}: {v:.3f}" for k, v in summary["action_distribution"].items()))
    return "\n".join(lines)