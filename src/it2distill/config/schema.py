"""Typed, immutable-ish experiment configuration (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    env_id: str = "CartPole-v1"
    n_envs: int = 8
    # Observation dims that receive noise (None = all). Use it to exclude binary/categorical
    # features, e.g. LunarLander leg-contact flags (dims 6, 7).
    noise_dims: tuple[int, ...] | None = None


@dataclass
class TeacherConfig:
    algo: str = "PPO"
    total_timesteps: int = 300_000
    target_reward: float = 500.0
    n_eval_episodes: int = 20                 # in-training EvalCallback (only for model selection)
    hyperparams: dict = field(default_factory=dict)
    # --- multi-seed teachers (protocol: >= 3 independent teachers) ---
    seeds: tuple[int, ...] = (0, 1, 2)
    # --- independent post-hoc verification (scripts/07_eval_teacher.py) ---
    verify_episodes: int = 50                 # episodes per eval seed
    verify_eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)  # disjoint from training seeds
    verify_min_mean: float = 0.0              # 0 -> derived as 0.98 * target_reward


@dataclass
class DataConfig:
    n_steps: int = 50_000
    teacher_deterministic: bool = True
    # Behaviour mix (preset name, see data/behaviors.py): clean | eps | noisy | mixed
    behavior_mix: str = "mixed"
    collection_epsilon: float = 0.1           # 'epsilon' behaviour: P(random action)
    # 'noisy' behaviour: teacher acts on obs + sigma_rel*std*z (sigma drawn per episode);
    # the stored observation is CLEAN and the label is the teacher action on the CLEAN state.
    behavior_noise_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0)
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)   # fractions of EPISODES (train, val, test)
    save_soft_targets: bool = True            # teacher action probabilities (discrete) for KL distillation
    obs_std_episodes: int = 30                # episodes used to estimate the clean-state std
    dagger_rounds: int = 0  # 0 = pure behavioural cloning


@dataclass
class FuzzyConfig:
    membership: str = "it2_uncertain_sigma"   # gaussian | it2_uncertain_sigma | it2_uncertain_mean
    tnorm: str = "product"                    # product | min
    reducer: str = "nie_tan"                  # weighted_average | nie_tan | bounds_mixing
    n_rules: int = 8
    consequent_order: int = 1                 # 0 or 1 (TSK)
    initializer: str = "kmeans"               # kmeans | random | teacher_tree
    fou_init: float = 0.3                     # initial relative FOU width
    fou_learnable: bool = True


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 0.0
    loss: str = "auto"                        # auto -> CE (discrete) / MSE (continuous); or kl_soft
    lambda_fou: float = 0.0                   # FOU regulariser
    lambda_sparsity: float = 0.0              # rule-sparsity regulariser
    lambda_overlap: float = 0.0               # MF-overlap (readability) regulariser
    train_noise_sigma: float = 0.0            # noise augmentation (relative sigma)
    patience: int = 20


@dataclass
class EvalConfig:
    noise_sigmas: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)  # relative to obs std
    n_episodes: int = 30
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    bootstrap_resamples: int = 10_000


@dataclass
class ExperimentConfig:
    name: str = "cartpole_it2"
    out_dir: str = "results"
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    data: DataConfig = field(default_factory=DataConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)