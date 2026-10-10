"""Aggregate CSVs -> tables (LaTeX) + figures + rule reports. TODO(WP8)."""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
