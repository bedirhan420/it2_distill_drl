"""CLI entry: thin wrapper over ExperimentPipeline.evaluate (no logic here)."""
import argparse

from it2distill.config.loader import load_config
from it2distill.experiments.pipeline import ExperimentPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pipe = ExperimentPipeline(cfg)
    raise SystemExit("use pipeline.run_all_seeds() or ExperimentPipeline.evaluate(students)")


if __name__ == "__main__":
    main()
