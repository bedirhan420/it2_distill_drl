import pytest

from it2distill.config.loader import load_config


def test_load_with_base_and_overrides():
    cfg = load_config("configs/env/cartpole.yaml", {"fuzzy.n_rules": 16, "train.lambda_fou": 0.01})
    assert cfg.env.env_id == "CartPole-v1"
    assert cfg.fuzzy.n_rules == 16 and cfg.train.lambda_fou == 0.01
    assert cfg.fuzzy.membership == "it2_uncertain_sigma"  # inherited from base.yaml


def test_teacher_seeds_and_noise_dims():
    cart = load_config("configs/env/cartpole.yaml")
    assert list(cart.teacher.seeds) == [0, 1, 2] and cart.env.noise_dims is None
    lunar = load_config("configs/env/lunarlander.yaml")
    assert list(lunar.env.noise_dims) == [0, 1, 2, 3, 4, 5]
    assert lunar.teacher.verify_min_mean == 200.0


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("teacher: {seedz: [0]}\n")
    with pytest.raises(KeyError):
        load_config(bad)


def test_loader_never_evaluates_annotations():
    """Regression: `X | None` annotations must not be evaluated (breaks on Python 3.9)."""
    import inspect

    from it2distill.config import loader
    assert "get_type_hints" not in inspect.getsource(loader).split('"""')[2]


def test_ablation_expand():
    from it2distill.experiments.ablation import AblationSpec
    spec = AblationSpec("x", {"a.b": [1, 2], "c.d": ["u", "v"]})
    assert len(spec.expand()) == 4