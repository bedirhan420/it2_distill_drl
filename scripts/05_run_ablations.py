import argparse

from it2distill.experiments.runner import run_ablations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default="results/ablations.csv")
    a = ap.parse_args()
    run_ablations(a.config, a.ablation, a.only).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
