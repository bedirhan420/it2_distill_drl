ENV ?= cartpole
SEED ?= 0
CFG := configs/env/$(ENV).yaml
.PHONY: install test lint teacher teacher-one teacher-force verify-teacher verify-teacher-noise data distill noise ablate report wp1
install: ; pip install -e ".[dev]"
test: ; pytest -q
lint: ; ruff check src tests scripts
teacher: ; python scripts/01_train_teacher.py --config $(CFG) --all-seeds
teacher-one: ; python scripts/01_train_teacher.py --config $(CFG) --seed $(SEED)
teacher-force: ; python scripts/01_train_teacher.py --config $(CFG) --all-seeds --force
verify-teacher: ; python scripts/07_eval_teacher.py --config $(CFG) --strict
verify-teacher-noise: ; python scripts/07_eval_teacher.py --config $(CFG) --noise --strict
wp1: teacher verify-teacher-noise
data: ; python scripts/02_collect_data.py --config $(CFG)
distill: ; python scripts/03_distill.py --config $(CFG)
noise: ; python scripts/04_evaluate_noise.py --config $(CFG)
ablate: ; python scripts/05_run_ablations.py --config $(CFG) --ablation configs/ablation/all.yaml
report: ; python scripts/06_make_report.py --results results/$(ENV)