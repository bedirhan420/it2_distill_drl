import numpy as np
import pytest

from it2distill.core.registry import Registry
from it2distill.evaluation.metrics import fidelity_discrete, relative_degradation
from it2distill.evaluation.pareto import pareto_front
from it2distill.evaluation.statistics import holm_correction


def test_registry_duplicate_and_unknown():
    r = Registry("x")
    r.register("a")(dict)
    with pytest.raises(KeyError):
        r.register("a")(dict)
    with pytest.raises(KeyError):
        r.create("b")


def test_noise_wrapper_zero_sigma_is_identity():
    gym = pytest.importorskip("gymnasium")
    from it2distill.envs.wrappers import GaussianObservationNoise
    env = GaussianObservationNoise(gym.make("CartPole-v1"), 0.0, np.ones(4), seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)


def test_metrics():
    assert fidelity_discrete(np.array([1, 0, 1]), np.array([1, 1, 1])) == pytest.approx(2 / 3)
    assert relative_degradation({0.0: 100.0, 0.1: 80.0})[0.1] == pytest.approx(0.2)


def test_pareto():
    pts = np.array([[2, 0.9], [4, 0.95], [4, 0.8]])  # (rules min, fidelity max)
    assert pareto_front(pts, (True, False)).tolist() == [True, True, False]


def test_holm():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and max(adj) <= 1.0
