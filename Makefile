UV ?= uv

TRAIN_LINEAR_CONFIG ?= experiments/linear_gaussian/01_supervised_edge_mlp.yaml
EVALUATE_LINEAR_CONFIG ?= experiments/linear_gaussian/00_oracle_check.yaml
NONLINEAR_CONFIG ?= experiments/nonlinear/01_sine_observation.yaml
RUN_DIR ?= outputs/latest

.PHONY: help setup lock test lint format check train-linear evaluate-linear plot-linear train-nonlinear evaluate-nonlinear clean

help:
	@printf "Targets:\n"
	@printf "  setup              Create/update the uv environment\n"
	@printf "  lock               Resolve and update uv.lock\n"
	@printf "  test               Run tests\n"
	@printf "  lint               Run ruff checks\n"
	@printf "  format             Format Python files with ruff\n"
	@printf "  check              Run tests and lint\n"
	@printf "  train-linear       Run linear-Gaussian training\n"
	@printf "  evaluate-linear    Run linear-Gaussian evaluation\n"
	@printf "  plot-linear        Plot linear-Gaussian results\n"
	@printf "  train-nonlinear    Run nonlinear training\n"
	@printf "  evaluate-nonlinear Run nonlinear evaluation\n"
	@printf "  clean              Remove local caches\n"

setup:
	$(UV) sync --dev

lock:
	$(UV) lock

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

check: test lint

train-linear:
	$(UV) run python scripts/train_linear_gaussian.py --config $(TRAIN_LINEAR_CONFIG)

evaluate-linear:
	$(UV) run python scripts/evaluate_linear_gaussian.py --config $(EVALUATE_LINEAR_CONFIG)

plot-linear:
	$(UV) run python scripts/plot_linear_gaussian.py --run-dir $(RUN_DIR)

train-nonlinear:
	$(UV) run python scripts/train_nonlinear.py --config $(NONLINEAR_CONFIG)

evaluate-nonlinear:
	$(UV) run python scripts/evaluate_nonlinear.py --config $(NONLINEAR_CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
