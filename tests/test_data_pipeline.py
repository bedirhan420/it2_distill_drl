import numpy as np
import pytest

pytest.importorskip("gymnasium")

from it2distill.config.schema import DataConfig  # noqa: E402
from it2distill.data.behaviors import BEHAVIOR_PRESETS, SOURCE_CODES, resolve_mix  # noqa: E402
from it2distill.data.collector import TeacherDataCollector  # noqa: E402
from it2distill.data.dataset import TransitionData, split_by_episode  # noqa: E402
from it2distill.data.diagnostics import summarize  # noqa: E402
from it2distill.data.repository import is_collected, load_splits, save_splits  # noqa: E402


class HeuristicCartPole:
    """Cheap stand-in teacher (no SB3 needed): push towards the side the pole leans."""

    def act(self, obs):
        return int(obs[2] + obs[3] > 0)

    def act_batch(self, obs):
        return (obs[:, 2] + obs[:, 3] > 0).astype(np.int64)


def _cfg(**kw):
    base = dict(n_steps=3000, behavior_mix="mixed", collection_epsilon=0.2,
                behavior_noise_sigmas=(1.0, 2.0), obs_std_episodes=5, save_soft_targets=False)
    base.update(kw)
    return DataConfig(**base)


def _collect(seed=0, **kw):
    return TeacherDataCollector("CartPole-v1", HeuristicCartPole(), _cfg(**kw), seed).collect()


def test_label_is_teacher_action_on_clean_obs_for_every_source():
    data = _collect()
    teacher = HeuristicCartPole()
    expected = np.array([teacher.act(o) for o in data.obs])
    assert np.array_equal(data.actions, expected)          # the core invariant
    assert set(np.unique(data.source)) == {SOURCE_CODES["clean"], SOURCE_CODES["epsilon"],
                                           SOURCE_CODES["noisy"]}


def test_quotas_roughly_respected_and_all_sources_present():
    data = _collect()
    steps = {c: int((data.source == c).sum()) for c in np.unique(data.source)}
    assert steps[0] >= 0.4 * 3000 and steps[1] >= 0.3 * 3000 and steps[2] >= 0.3 * 3000


def test_collection_is_deterministic_and_seed_dependent():
    a, b, c = _collect(seed=0), _collect(seed=0), _collect(seed=1)
    assert np.array_equal(a.obs, b.obs) and np.array_equal(a.actions, b.actions)
    assert not np.array_equal(a.obs[:50], c.obs[:50])


def test_epsilon_and_noisy_widen_state_coverage():
    data = _collect(n_steps=6000, collection_epsilon=0.3)
    per = summarize(data)["per_source"]
    assert per["epsilon"]["state_spread_vs_clean"] != pytest.approx(1.0, abs=1e-3)
    assert "noisy" in per and per["noisy"]["mean_episode_len"] > 0


def test_split_is_episode_disjoint_stratified_and_complete():
    data = _collect(n_steps=6000)
    sp = split_by_episode(data, (0.7, 0.15, 0.15), seed=0)
    ids = [set(np.unique(s.episode_id).tolist()) for s in (sp.train, sp.val, sp.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(sp.sizes().values()) == len(data)
    for s in (sp.train, sp.val, sp.test):
        assert set(np.unique(s.source)) == set(np.unique(data.source))   # every split sees every source


def test_split_rejects_too_few_episodes():
    obs = np.zeros((4, 4), dtype=np.float32)
    data = TransitionData(obs, np.zeros(4, dtype=np.int64), np.array([0, 0, 1, 1]), np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError):
        split_by_episode(data, (0.7, 0.15, 0.15), 0)


def test_concat_keeps_episode_ids_unique():
    a, b = _collect(n_steps=1000, behavior_mix="clean"), _collect(n_steps=1000, behavior_mix="clean", seed=1)
    both = TransitionData.concat([a, b])
    assert both.n_episodes == a.n_episodes + b.n_episodes and len(both) == len(a) + len(b)


def test_dagger_round_labels_are_teacher_actions():
    collector = TeacherDataCollector("CartPole-v1", HeuristicCartPole(), _cfg(), 0)
    student = type("Const", (), {"act": lambda self, o: 0, "act_batch": lambda self, o: np.zeros(len(o))})()
    d = collector.collect_dagger(student, n_steps=800, beta=0.0, round_idx=0, noise_sigma=1.0)
    assert np.array_equal(d.actions, np.array([HeuristicCartPole().act(o) for o in d.obs]))
    assert set(np.unique(d.source)) == {SOURCE_CODES["dagger"]}
    with pytest.raises(ValueError):
        collector.collect_dagger(student, 100, 0.5, round_idx=5)


def test_presets_valid_and_repository_roundtrip(tmp_path):
    for name in BEHAVIOR_PRESETS:
        assert abs(sum(resolve_mix(name).values()) - 1.0) < 1e-9
    with pytest.raises(KeyError):
        resolve_mix("nope")
    data = _collect()
    sp = split_by_episode(data, (0.7, 0.15, 0.15), 0)
    assert not is_collected(tmp_path)
    save_splits(sp, tmp_path, {"k": np.float32(1.5), "arr": np.arange(3)})
    loaded, meta = load_splits(tmp_path)
    assert is_collected(tmp_path) and meta["k"] == 1.5
    assert np.array_equal(loaded.train.obs, sp.train.obs) and loaded.val.soft_targets is None