import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from it2distill.envs.wrappers import GaussianObservationNoise  # noqa: E402


def test_noise_is_identical_across_envs_for_same_episode_seed():
    env_a = GaussianObservationNoise(gym.make("CartPole-v1"), 0.3, np.ones(4), seed=0)
    env_b = GaussianObservationNoise(gym.make("CartPole-v1"), 0.3, np.ones(4), seed=99)  # different init seed
    o_a, _ = env_a.reset(seed=5)
    o_b, _ = env_b.reset(seed=5)
    np.testing.assert_allclose(o_a, o_b)                 # reset(seed) fully determines the noise
    a2, *_ = env_a.step(0)
    b2, *_ = env_b.step(1)                               # different actions, same noise draw
    clean = gym.make("CartPole-v1")
    clean.reset(seed=5)
    ca, *_ = clean.step(0)
    clean.reset(seed=5)
    cb, *_ = clean.step(1)
    np.testing.assert_allclose(a2 - ca, b2 - cb, atol=1e-5)


def test_zero_sigma_is_identity():
    env = GaussianObservationNoise(gym.make("CartPole-v1"), 0.0, np.ones(4), seed=0)
    clean = gym.make("CartPole-v1")
    np.testing.assert_allclose(env.reset(seed=1)[0], clean.reset(seed=1)[0])


def test_noise_dims_leaves_other_dims_untouched():
    env = GaussianObservationNoise(gym.make("CartPole-v1"), 0.5, np.ones(4), seed=0, noise_dims=[0, 1])
    clean = gym.make("CartPole-v1")
    noisy, _ = env.reset(seed=3)
    ref, _ = clean.reset(seed=3)
    np.testing.assert_allclose(noisy[2:], ref[2:], atol=1e-6)
    assert not np.allclose(noisy[:2], ref[:2])


def test_noise_dims_out_of_range():
    with pytest.raises(ValueError):
        GaussianObservationNoise(gym.make("CartPole-v1"), 0.5, np.ones(4), noise_dims=[4])