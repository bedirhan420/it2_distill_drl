"""WP2 - collect teacher state-action data, split by episode, save to disk.

Examples:
    python scripts/02_collect_data.py --config configs/env/cartpole.yaml --all-seeds
    python scripts/02_collect_data.py --config configs/env/cartpole.yaml --seed 1 --force

Output per teacher seed: results/<name>/teacher_s<seed>/data/{train,val,test}.npz + data_meta.json

Dataset invariant: stored obs are CLEAN; labels are the teacher's action on that clean obs. The
'epsilon' and 'noisy' behaviours only change which states get visited (coverage).
"""
from __future__ import annotations

import argparse
import dataclasses

from it2distill.config.loader import load_config
from it2distill.data.behaviors import resolve_mix
from it2distill.data.collector import TeacherDataCollector
from it2distill.data.dataset import split_by_episode
from it2distill.data.diagnostics import render, summarize
from it2distill.data.repository import data_dir, is_collected, save_splits
from it2distill.envs.factory import action_spec_of, make_env
from it2distill.teacher.adapter import SB3TeacherPolicy
from it2distill.teacher.paths import best_model_path, is_trained, teacher_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="collect for a single teacher seed")
    ap.add_argument("--all-seeds", action="store_true", help="collect for every seed in teacher.seeds")
    ap.add_argument("--force", action="store_true", help="re-collect even if a dataset exists")
    return ap.parse_args()


def collect_for_teacher(cfg, seed: int, force: bool) -> None:
    t_dir = teacher_dir(cfg.out_dir, cfg.name, seed)
    out = data_dir(t_dir)
    if not is_trained(t_dir):
        print(f"[teacher {seed}] NOT TRAINED -> run `make teacher` first")
        return
    if is_collected(out) and not force:
        print(f"[teacher {seed}] dataset exists -> {out} (use --force to re-collect)")
        return

    model_path = best_model_path(t_dir)
    teacher = SB3TeacherPolicy(model_path, deterministic=cfg.data.teacher_deterministic)
    collector = TeacherDataCollector(cfg.env.env_id, teacher, cfg.data, seed, cfg.env.noise_dims)
    print(f"[teacher {seed}] collecting {cfg.data.n_steps:,} steps, mix='{cfg.data.behavior_mix}' ...")
    data = collector.collect()
    splits = split_by_episode(data, cfg.data.splits, seed)

    env = make_env(cfg.env.env_id)
    spec, obs_dim = action_spec_of(env), int(env.observation_space.shape[0])
    env.close()

    summary = summarize(data)
    meta = {
        "teacher_seed": seed, "teacher_model": str(model_path), "env_id": cfg.env.env_id,
        "obs_dim": obs_dim, "action_kind": spec.kind, "action_dim": spec.dim,
        "obs_std": collector.obs_std, "noise_dims": cfg.env.noise_dims,
        "behavior_mix": resolve_mix(cfg.data.behavior_mix), "data_config": dataclasses.asdict(cfg.data),
        "split_sizes_steps": splits.sizes(),
        "split_episodes": {k: getattr(splits, k).n_episodes for k in ("train", "val", "test")},
        "has_soft_targets": data.soft_targets is not None, "summary": summary,
    }
    save_splits(splits, out, meta)
    print(render(summary))
    print(f"  splits (steps): {splits.sizes()}  episodes: {meta['split_episodes']}")
    print(f"  saved -> {out}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = list(cfg.teacher.seeds) if args.all_seeds else [args.seed if args.seed is not None else cfg.seed]
    for seed in seeds:
        collect_for_teacher(cfg, seed, args.force)


if __name__ == "__main__":
    main()