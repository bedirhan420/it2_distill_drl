import pytest
import torch

from it2distill.fuzzy.membership import MF_REGISTRY


@pytest.mark.parametrize("name", ["gaussian", "it2_uncertain_sigma", "it2_uncertain_mean"])
def test_shapes_and_bounds(name):
    mf = MF_REGISTRY.create(name, n_rules=5, n_inputs=3, fou_init=0.4)
    out = mf(torch.randn(7, 3))
    assert out.lower.shape == out.upper.shape == (7, 5, 3)
    assert torch.all(out.lower <= out.upper + 1e-6), "lower MF must not exceed upper MF"
    assert torch.all((out.lower >= 0) & (out.upper <= 1 + 1e-6))


def test_it2_collapses_to_t1_when_fou_vanishes():
    t1 = MF_REGISTRY.create("gaussian", n_rules=4, n_inputs=2)
    it2 = MF_REGISTRY.create("it2_uncertain_sigma", n_rules=4, n_inputs=2, fou_init=1e-6)
    it2.load_state_dict({**it2.state_dict(), "centers": t1.centers.data, "log_sigma": t1.log_sigma.data})
    x = torch.randn(10, 2)
    assert torch.allclose(it2(x).lower, t1(x).lower, atol=1e-4)
    assert torch.allclose(it2(x).upper, t1(x).lower, atol=1e-3)
