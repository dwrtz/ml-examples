UV ?= uv

TRAIN_LINEAR_CONFIG ?= experiments/linear_gaussian/01_supervised_edge_mlp.yaml
TRAIN_LINEAR_ELBO_CONFIG ?= experiments/linear_gaussian/02_elbo_edge_mlp.yaml
EVALUATE_LINEAR_CONFIG ?= experiments/linear_gaussian/00_oracle_check.yaml
NONLINEAR_CONFIG ?= experiments/nonlinear/01_sine_observation.yaml
RUN_DIR ?= outputs/linear_gaussian_supervised_edge_mlp
RUN_DIR_ELBO ?= outputs/linear_gaussian_elbo_edge_mlp
LINEAR_COMPARISON ?= outputs/linear_gaussian_comparison.md
LINEAR_SWEEP_DIR ?= outputs/linear_gaussian_sweep
LINEAR_SWEEP_SEEDS ?= 321,322,323,324,325
ELBO_ABLATION_DIR ?= outputs/linear_gaussian_elbo_ablation
ELBO_ABLATION_SAMPLES ?= 1,4,8,16,32
ELBO_ABLATION_STEPS ?= 250,1000
EDGE_REGULARIZER_DIR ?= outputs/linear_gaussian_edge_regularizer
EDGE_REGULARIZER_WEIGHTS ?= 0,0.01,0.05,0.1
TRANSITION_CONSISTENCY_DIR ?= outputs/linear_gaussian_transition_consistency
TRANSITION_CONSISTENCY_WEIGHTS ?= 0,0.01,0.05,0.1

.PHONY: help setup lock test lint format check train-linear train-linear-elbo evaluate-linear plot-linear plot-linear-elbo compare-linear sweep-linear sweep-elbo-ablation sweep-edge-regularizer sweep-transition-consistency train-nonlinear evaluate-nonlinear clean

help:
	@printf "Targets:\n"
	@printf "  setup              Create/update the uv environment\n"
	@printf "  lock               Resolve and update uv.lock\n"
	@printf "  test               Run tests\n"
	@printf "  lint               Run ruff checks\n"
	@printf "  format             Format Python files with ruff\n"
	@printf "  check              Run tests and lint\n"
	@printf "  train-linear       Run linear-Gaussian training\n"
	@printf "  train-linear-elbo  Run ELBO linear-Gaussian training\n"
	@printf "  evaluate-linear    Run linear-Gaussian evaluation\n"
	@printf "  plot-linear        Plot linear-Gaussian results\n"
	@printf "  plot-linear-elbo   Plot ELBO linear-Gaussian results\n"
	@printf "  compare-linear     Compare supervised and ELBO linear-Gaussian runs\n"
	@printf "  sweep-linear       Train and aggregate linear-Gaussian seed sweep\n"
	@printf "  sweep-elbo-ablation Run ELBO MC-sample/training-budget ablation\n"
	@printf "  sweep-edge-regularizer Run diagnostic oracle edge-KL regularizer sweep\n"
	@printf "  sweep-transition-consistency Run unsupervised transition regularizer sweep\n"
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

train-linear-elbo:
	$(UV) run python scripts/train_linear_gaussian.py --config $(TRAIN_LINEAR_ELBO_CONFIG)

evaluate-linear:
	$(UV) run python scripts/evaluate_linear_gaussian.py --config $(EVALUATE_LINEAR_CONFIG)

plot-linear:
	$(UV) run python scripts/plot_linear_gaussian.py --run-dir $(RUN_DIR)

plot-linear-elbo:
	$(UV) run python scripts/plot_linear_gaussian.py --run-dir $(RUN_DIR_ELBO)

compare-linear:
	$(UV) run python scripts/compare_linear_gaussian.py --supervised-run-dir $(RUN_DIR) --elbo-run-dir $(RUN_DIR_ELBO) --output $(LINEAR_COMPARISON)

sweep-linear:
	$(UV) run python scripts/sweep_linear_gaussian.py --seeds $(LINEAR_SWEEP_SEEDS) --output-dir $(LINEAR_SWEEP_DIR)

sweep-elbo-ablation:
	$(UV) run python scripts/sweep_elbo_ablation.py --seeds $(LINEAR_SWEEP_SEEDS) --samples $(ELBO_ABLATION_SAMPLES) --steps $(ELBO_ABLATION_STEPS) --output-dir $(ELBO_ABLATION_DIR)

sweep-edge-regularizer:
	$(UV) run python scripts/sweep_edge_regularizer.py --seeds $(LINEAR_SWEEP_SEEDS) --weights $(EDGE_REGULARIZER_WEIGHTS) --output-dir $(EDGE_REGULARIZER_DIR)

sweep-transition-consistency:
	$(UV) run python scripts/sweep_transition_consistency.py --seeds $(LINEAR_SWEEP_SEEDS) --weights $(TRANSITION_CONSISTENCY_WEIGHTS) --output-dir $(TRANSITION_CONSISTENCY_DIR)

train-nonlinear:
	$(UV) run python scripts/train_nonlinear.py --config $(NONLINEAR_CONFIG)

evaluate-nonlinear:
	$(UV) run python scripts/evaluate_nonlinear.py --config $(NONLINEAR_CONFIG)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
