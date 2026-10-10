"""WP1 - train PPO teacher(s). One independent teacher per seed.

Examples:
    python scripts/01_train_teacher.py --config configs/env/cartpole.yaml --all-seeds
    python scripts/01_train_teacher.py --config configs/env/cartpole.yaml --seed 1
    python scripts/01_train_teacher.py --config configs/env/cartpole.yaml --all-seeds --force

Output per seed: results/<name>/teacher_s<seed>/{best_model.zip, teacher_final.zip,
                 evaluations.npz, teacher_meta.json}
"""
from __future__ import annotations

import argparse

from it2distill.config.loader import load_config
from it2distill.teacher.paths import best_model_path, is_trained, teacher_dir
from it2distill.teacher.trainer import TeacherTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="train a single teacher with this seed")
    ap.add_argument("--all-seeds", action="store_true", help="train every seed in teacher.seeds")
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run exists")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.all_seeds:
        seeds = list(cfg.teacher.seeds)
    else:
        seeds = [args.seed if args.seed is not None else cfg.seed]

    for seed in seeds:
        out_dir = teacher_dir(cfg.out_dir, cfg.name, seed)
        if is_trained(out_dir) and not args.force:
            print(f"[{cfg.name}] teacher seed={seed}: already trained -> {out_dir} (use --force to retrain)")
            continue
        print(f"[{cfg.name}] training teacher seed={seed} (budget={cfg.teacher.total_timesteps:,} steps)")
        best = TeacherTrainer(cfg.env, cfg.teacher, out_dir, seed).train()
        print(f"[{cfg.name}] teacher seed={seed} done -> {best}")


if __name__ == "__main__":
    main()