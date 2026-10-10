"""Figure generation for the paper (matplotlib only; one function per figure)."""
from __future__ import annotations


def plot_return_vs_sigma(df, path): raise NotImplementedError          # Fig: degradation curves + CI
def plot_pareto(df, path): raise NotImplementedError                   # Fig: rules vs robustness
def plot_membership_functions(model, path): raise NotImplementedError  # Fig: T1 vs IT2 MFs with FOU
def plot_ablation_heatmap(df, path): raise NotImplementedError         # Fig: ablation grid
