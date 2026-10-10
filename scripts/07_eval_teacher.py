"""WP1 - independent verification of the trained teachers (run BEFORE distillation).

Why: the in-training EvalCallback (20 episodes, best-checkpoint selection) is optimistically biased
("winner's curse"). This script re-evaluates every teacher on fresh evaluation seeds and episodes.

Examples:
    python scripts/07_eval_teacher.py --config configs/env/cartpole.yaml
    python scripts/07_eval_teacher.py --config configs/env/cartpole.yaml --noise
    python scripts/07_eval_teacher.py --config configs/env/lunarlander.yaml --noise --noise-seeds 3 --noise-episodes 10
    python scripts/07_eval_teacher.py --config configs/env/cartpole.yaml --teachers 0 1 --strict

Normalised return = (R - R_random) / (R_teacher_clean - R_random)   (0 = random, 1 = teacher).
It is environment-agnostic (works for negative-return envs); robustness_auc uses the same scale.

Outputs (results/<name>/):
    teacher_verification.csv   per (teacher, eval_seed) clean-return statistics
    teacher_verification.json  per-teacher pooled summary, PASS/FAIL, (--noise) robustness AUC
    teacher_noise_curve.csv    (--noise) return + normalised return per (teacher, sigma, eval_seed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from it2distill.config.loader import load_config
from it2distill.evaluation.metrics import normalize_return, robustness_auc
from it2distill.evaluation.rollout import collect_obs_std, evaluate_returns, random_baseline_return
from it2distill.evaluation.statistics import bootstrap_ci
from it2distill.teacher.adapter import SB3TeacherPolicy
from it2distill.teacher.paths import best_model_path, is_trained, teacher_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--teachers", type=int, nargs="*", default=None, help="teacher seeds (default: cfg)")
    ap.add_argument("--noise", action="store_true", help="also measure the PPO noise-collapse curve")
    ap.add_argument("--noise-seeds", type=int, default=5, help="# eval seeds used for the noise curve")
    ap.add_argument("--noise-episodes", type=int, default=None, help="episodes per (sigma, seed)")
    ap.add_argument("--random-episodes", type=int, default=100, help="episodes for the random baseline")
    ap.add_argument("--strict", action="store_true", help="exit with code 1 if any teacher fails")
    return ap.parse_args()


def verify_clean(policy, cfg, teacher_seed: int) -> tuple[list[dict], np.ndarray]:
    rows, pooled = [], []
    for ev_seed in cfg.teacher.verify_eval_seeds:
        r = evaluate_returns(policy, cfg.env.env_id, 0.0, None, cfg.teacher.verify_episodes, ev_seed)
        pooled.append(r)
        rows.append({
            "teacher_seed": teacher_seed, "eval_seed": ev_seed, "n_episodes": len(r),
            "mean": r.mean(), "std": r.std(ddof=1), "min": r.min(),
            "frac_at_target": float(np.mean(r >= cfg.teacher.target_reward)),
        })
    return rows, np.concatenate(pooled)


def noise_curve(policy, cfg, teacher_seed: int, n_seeds: int, n_episodes: int,
                r_random: float, r_ref: float) -> tuple[list[dict], dict]:
    """Noise sweep. Returns tidy rows and a summary dict (normalised curve + robustness AUC)."""
    obs_std = collect_obs_std(policy, cfg.env.env_id, n_episodes=50, seed=999)
    noise_dims = cfg.env.noise_dims
    rows = []
    for sigma in cfg.eval.noise_sigmas:
        for ev_seed in cfg.eval.seeds[:n_seeds]:
            r = evaluate_returns(policy, cfg.env.env_id, sigma, obs_std, n_episodes, ev_seed, noise_dims)
            rows.append({"teacher_seed": teacher_seed, "sigma": sigma, "eval_seed": ev_seed,
                         "mean_return": r.mean(),
                         "normalized_return": normalize_return(r.mean(), r_random, r_ref)})
        sub = [x for x in rows if x["sigma"] == sigma]
        print(f"    sigma={sigma:<5} return={np.mean([x['mean_return'] for x in sub]):9.1f}   "
              f"normalized={np.mean([x['normalized_return'] for x in sub]):6.3f}")

    by_sigma = {s: float(np.mean([x["mean_return"] for x in rows if x["sigma"] == s]))
                for s in cfg.eval.noise_sigmas}
    auc = robustness_auc(by_sigma, r_random, r_ref)
    print(f"    robustness_auc = {auc:.3f}")
    summary = {"obs_std": obs_std.tolist(), "robustness_auc": auc, "r_random": r_random, "r_ref": r_ref,
               "noise_dims": noise_dims,
               "normalized_by_sigma": {str(s): normalize_return(v, r_random, r_ref)
                                       for s, v in by_sigma.items()}}
    return rows, summary


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.out_dir) / cfg.name
    seeds = args.teachers if args.teachers is not None else list(cfg.teacher.seeds)
    threshold = cfg.teacher.verify_min_mean or 0.98 * cfg.teacher.target_reward
    n_noise_eps = args.noise_episodes or cfg.eval.n_episodes

    r_random = None
    if args.noise:
        r_random = random_baseline_return(cfg.env.env_id, args.random_episodes)
        print(f"[baseline] random-policy mean return = {r_random:.1f} ({args.random_episodes} episodes)")

    clean_rows, noise_rows, summary = [], [], {}
    all_pass = True
    for ts in seeds:
        d = teacher_dir(cfg.out_dir, cfg.name, ts)
        if not is_trained(d):
            print(f"[teacher {ts}] NOT TRAINED ({d}) -> run `make teacher`")
            summary[str(ts)] = {"verdict": "MISSING"}
            all_pass = False
            continue
        model_path = best_model_path(d)
        print(f"[teacher {ts}] verifying {model_path}")
        policy = SB3TeacherPolicy(model_path)

        rows, pooled = verify_clean(policy, cfg, ts)
        clean_rows += rows
        mean, lo, hi = bootstrap_ci(pooled, cfg.eval.bootstrap_resamples)
        passed = mean >= threshold
        all_pass &= passed
        summary[str(ts)] = {
            "model": str(model_path), "n_episodes": int(len(pooled)),
            "mean": mean, "ci95": [lo, hi], "std": float(pooled.std(ddof=1)),
            "min": float(pooled.min()), "frac_at_target": float(np.mean(pooled >= cfg.teacher.target_reward)),
            "threshold": threshold, "verdict": "PASS" if passed else "FAIL",
        }
        print(f"    clean: mean={mean:.1f} CI95=[{lo:.1f},{hi:.1f}] min={pooled.min():.0f} "
              f"-> {summary[str(ts)]['verdict']} (threshold {threshold:.1f})")

        if args.noise:
            print("    noise sweep (sigma relative to the teacher's clean-state std):")
            rows_n, noise_summary = noise_curve(policy, cfg, ts, args.noise_seeds, n_noise_eps,
                                                r_random, mean)  # r_ref = this teacher's clean mean
            noise_rows += rows_n
            summary[str(ts)]["noise"] = noise_summary

    out.mkdir(parents=True, exist_ok=True)
    if clean_rows:
        pd.DataFrame(clean_rows).to_csv(out / "teacher_verification.csv", index=False)
    (out / "teacher_verification.json").write_text(json.dumps(summary, indent=2))
    if noise_rows:
        df = pd.DataFrame(noise_rows)
        df.to_csv(out / "teacher_noise_curve.csv", index=False)
        print("\nNormalized return by teacher x sigma (mean over eval seeds):")
        print(df.pivot_table(index="teacher_seed", columns="sigma", values="normalized_return").round(3))

    print("\nVERDICT:", "ALL TEACHERS PASS" if all_pass else "SOME TEACHERS FAILED / MISSING")
    return 0 if (all_pass or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())