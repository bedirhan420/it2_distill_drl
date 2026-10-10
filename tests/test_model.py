import numpy as np
import torch

from it2distill.config.schema import FuzzyConfig
from it2distill.core.types import ActionSpec, Standardization
from it2distill.fuzzy.builder import FuzzyPolicyBuilder


def _build(membership, reducer):
    std = Standardization(np.zeros(4), np.ones(4))
    x = np.random.randn(500, 4)
    y = np.random.randint(0, 2, 500)
    cfg = FuzzyConfig(membership=membership, reducer=reducer, n_rules=6)
    return FuzzyPolicyBuilder.from_config(cfg, 4, ActionSpec("discrete", 2), std, x, y)


def test_forward_shapes_and_gradients():
    for mem, red in [("gaussian", "weighted_average"), ("it2_uncertain_sigma", "nie_tan"),
                     ("it2_uncertain_mean", "bounds_mixing")]:
        m = _build(mem, red)
        out = m(torch.randn(16, 4))
        assert out.shape == (16, 2)
        out.sum().backward()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_n_rules():
    assert _build("gaussian", "weighted_average").n_rules() == 6
