"""Pure metric functions (no I/O) -> trivially unit-testable.

Definitions (document in the paper):
  normalize_return    = (R - R_random) / (R_ref - R_random)
                        R_random: uniform-random policy, R_ref: clean teacher (both measured, per env).
                        0 = random, 1 = teacher; can exceed 1 or go below 0. Env-agnostic (works with
                        negative-return envs such as LunarLander), unlike R / R_max.
  fidelity_discrete   = P[ student(s~) == teacher(s_clean) ]   (s~ = noisy observation)
  fidelity_continuous = 1 - ||a_s - a_t||^2 / Var(a_t)         (R^2-style, clipped to [-inf, 1])
  robustness_auc      = trapezoid AUC of normalised return vs sigma, divided by the sigma range
                        (1.0 = teacher-level at every noise level; higher = more robust)
  relative_degradation(sigma) = 1 - R(sigma)/R(0)   (only meaningful for positive-return envs;
                        prefer normalised return elsewhere)
"""
from __future__ import annotations

import numpy as np


def normalize_return(r: float, r_random: float, r_ref: float) -> float:
    span = r_ref - r_random
    if span <= 0:
        raise ValueError(f"reference return ({r_ref}) must exceed random-policy return ({r_random})")
    return (r - r_random) / span


def fidelity_discrete(student_actions: np.ndarray, teacher_actions: np.ndarray) -> float:
    return float(np.mean(student_actions == teacher_actions))


def fidelity_continuous(student_actions: np.ndarray, teacher_actions: np.ndarray) -> float:
    var = np.var(teacher_actions, axis=0).sum() + 1e-12
    return float(1.0 - np.mean(np.sum((student_actions - teacher_actions) ** 2, axis=-1)) / var)


def relative_degradation(returns_by_sigma: dict[float, float]) -> dict[float, float]:
    base = returns_by_sigma[0.0]
    return {s: 1.0 - r / base for s, r in returns_by_sigma.items()}


def robustness_auc(returns_by_sigma: dict[float, float], r_min: float, r_max: float) -> float:
    """Normalised area under the return-vs-sigma curve (trapezoid rule, own implementation:
    np.trapz was removed in recent NumPy releases)."""
    sig = np.array(sorted(returns_by_sigma), dtype=float)
    if len(sig) < 2:
        raise ValueError("need at least two sigma levels")
    norm = np.array([normalize_return(returns_by_sigma[s], r_min, r_max) for s in sig])
    area = np.sum((norm[1:] + norm[:-1]) / 2.0 * np.diff(sig))
    return float(area / (sig[-1] - sig[0]))


def complexity_summary(n_rules: int, n_inputs: int, consequent_order: int, n_active_terms: int) -> dict:
    """Readability proxies: #rules, avg antecedent length (after don't-care pruning), #params."""
    return {"n_rules": n_rules, "avg_antecedent_len": n_active_terms / max(n_rules, 1),
            "n_consequent_params": n_rules * (1 + (n_inputs if consequent_order else 0))}