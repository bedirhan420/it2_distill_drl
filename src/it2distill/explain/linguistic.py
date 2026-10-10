"""Maps MF centers to linguistic terms (e.g. 'very low' ... 'very high') per input dimension
via quantiles of the TRAIN data; also names the input variables."""
from __future__ import annotations

import numpy as np

_TERMS = {
    3: ["low", "medium", "high"],
    5: ["very low", "low", "medium", "high", "very high"],
}


class LinguisticLabeler:
    def __init__(self, feature_names: list[str], x_train: np.ndarray, n_terms: int = 5) -> None:
        self.names, self.terms = feature_names, _TERMS[n_terms]
        qs = np.linspace(0, 1, n_terms + 1)[1:-1]
        self.edges = np.quantile(x_train, qs, axis=0)  # (n_terms-1, D)

    def label(self, dim: int, center: float) -> str:
        return self.terms[int(np.searchsorted(self.edges[:, dim], center))]
