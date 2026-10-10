import pytest

from it2distill.evaluation.metrics import normalize_return, robustness_auc


def test_normalize_return_positive_env():
    assert normalize_return(500, 20, 500) == pytest.approx(1.0)
    assert normalize_return(20, 20, 500) == pytest.approx(0.0)
    assert normalize_return(260, 20, 500) == pytest.approx(0.5)


def test_normalize_return_negative_returns_env():  # LunarLander-like
    assert normalize_return(-100, -150, 250) == pytest.approx(0.125)
    assert normalize_return(-150, -150, 250) == pytest.approx(0.0)


def test_normalize_requires_ref_above_random():
    with pytest.raises(ValueError):
        normalize_return(1, 10, 10)


def test_robustness_auc_bounds():
    flat = {0.0: 500.0, 0.2: 500.0, 0.5: 500.0}
    assert robustness_auc(flat, 20, 500) == pytest.approx(1.0)
    random_level = {0.0: 20.0, 0.5: 20.0}
    assert robustness_auc(random_level, 20, 500) == pytest.approx(0.0)
    linear_collapse = {0.0: 500.0, 0.5: 20.0}
    assert robustness_auc(linear_collapse, 20, 500) == pytest.approx(0.5)