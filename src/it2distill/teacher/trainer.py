"""Teacher (oracle) training with SB3 PPO. One directory per teacher seed; reproducible metadata."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env

from it2distill.config.schema import EnvConfig, TeacherConfig

from .paths import META_FILE, best_model_path

_RESERVED = {"seed", "device", "verbose"}  # controlled by the trainer, not by config


class TeacherTrainer:
    def __init__(self, env_cfg: EnvConfig, cfg: TeacherConfig, out_dir: Path, seed: int) -> None:
        self.env_cfg, self.cfg, self.out_dir, self.seed = env_cfg, cfg, Path(out_dir), seed

    def train(self) -> Path:
        """Train until the reward threshold or the timestep budget; return the path of the BEST model."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        venv = make_vec_env(self.env_cfg.env_id, n_envs=self.env_cfg.n_envs, seed=self.seed)
        eval_env = make_vec_env(self.env_cfg.env_id, n_envs=1, seed=self.seed + 10_000)

        stop = StopTrainingOnRewardThreshold(reward_threshold=self.cfg.target_reward, verbose=1)
        eval_cb = EvalCallback(
            eval_env,
            callback_on_new_best=stop,
            n_eval_episodes=self.cfg.n_eval_episodes,
            eval_freq=max(5_000 // self.env_cfg.n_envs, 1),
            best_model_save_path=str(self.out_dir),   # -> best_model.zip
            log_path=str(self.out_dir),               # -> evaluations.npz (learning curve)
            deterministic=True,
        )
        hyperparams = {k: v for k, v in self.cfg.hyperparams.items() if k not in _RESERVED}
        # CPU on purpose: small MLP policies are faster on CPU and CPU runs are reproducible.
        model = PPO("MlpPolicy", venv, seed=self.seed, device="cpu", verbose=0, **hyperparams)

        t0 = time.time()
        model.learn(self.cfg.total_timesteps, callback=eval_cb)
        elapsed = time.time() - t0

        model.save(self.out_dir / "teacher_final")
        best = best_model_path(self.out_dir)
        self._write_meta(model, eval_cb, elapsed, best)
        venv.close()
        eval_env.close()
        return best

    def _write_meta(self, model: PPO, cb: EvalCallback, elapsed: float, best: Path) -> None:
        best_mean = float(cb.best_mean_reward)
        meta = {
            "seed": self.seed,
            "env_id": self.env_cfg.env_id,
            "n_envs": self.env_cfg.n_envs,
            "target_reward": self.cfg.target_reward,
            "total_timesteps_budget": self.cfg.total_timesteps,
            "timesteps_used": int(model.num_timesteps),
            "reached_threshold": best_mean >= self.cfg.target_reward,
            "best_eval_mean_reward": best_mean,
            "n_eval_episodes": self.cfg.n_eval_episodes,
            "hyperparams": self.cfg.hyperparams,
            "best_model": best.name,
            "train_seconds": round(elapsed, 1),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "gymnasium": gymnasium.__version__,
            },
        }
        (self.out_dir / META_FILE).write_text(json.dumps(meta, indent=2))