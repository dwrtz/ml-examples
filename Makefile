UV ?= uv

TRAIN_LINEAR_CONFIG ?= experiments/linear_gaussian/01_supervised_edge_mlp.yaml
EVALUATE_LINEAR_CONFIG ?= experiments/linear_gaussian/00_oracle_check.yaml
NONLINEAR_CONFIG ?= experiments/nonlinear/01_sine_observation.yaml
TRAIN_NONLINEAR_CONFIG ?= experiments/nonlinear/08_direct_elbo_sine_mlp.yaml
RUN_DIR ?= outputs/linear_gaussian_supervised_edge_mlp

NONLINEAR_REFERENCE_DIR ?= outputs/nonlinear_reference_suite
NONLINEAR_REFERENCE_CONFIGS ?= experiments/nonlinear/01_sine_observation.yaml,experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml,experiments/nonlinear/05_zero_sine_observation.yaml,experiments/nonlinear/06_random_normal_sine_observation.yaml,experiments/nonlinear/15_tanh_observation.yaml,experiments/nonlinear/16_cubic_observation.yaml,experiments/nonlinear/17_heteroskedastic_observation.yaml
NONLINEAR_LEARNED_DIR ?= outputs/nonlinear_learned_suite
NONLINEAR_LEARNED_CONFIGS ?= $(NONLINEAR_REFERENCE_CONFIGS)
NONLINEAR_LEARNED_MODELS ?= direct_elbo,structured_elbo
NONLINEAR_LEARNED_STEPS ?= 250
NONLINEAR_LEARNED_SEEDS ?=
NONLINEAR_REFERENCE_VARIANCE_RATIO_WEIGHT ?=
NONLINEAR_REFERENCE_TIME_VARIANCE_RATIO_WEIGHT ?=
NONLINEAR_REFERENCE_LOW_OBSERVATION_VARIANCE_RATIO_WEIGHT ?=

NONLINEAR_SWEEP_METRICS ?= outputs/nonlinear_calibration_weight_sweep_250/w1/metrics.csv,outputs/nonlinear_calibration_weight_sweep_250/w3/metrics.csv,outputs/nonlinear_calibration_weight_sweep_250/w10/metrics.csv
NONLINEAR_SWEEP_BASELINE_METRICS ?= outputs/nonlinear_calibration_cached_250/metrics.csv
NONLINEAR_SWEEP_WEIGHTS ?= 1,3,10
NONLINEAR_SWEEP_PATTERNS ?= weak_sinusoidal,intermittent_sinusoidal
NONLINEAR_SWEEP_PLOT_DIR ?= outputs/nonlinear_calibration_weight_sweep_250

.PHONY: help setup lock test lint format check train-linear evaluate-linear plot-linear train-nonlinear evaluate-nonlinear sweep-nonlinear-reference sweep-nonlinear-learned plot-nonlinear plot-nonlinear-sweep clean

help:
	@printf "Targets:\n"
	@printf "  setup                       Create/update the uv environment\n"
	@printf "  lock                        Resolve and update uv.lock\n"
	@printf "  test                        Run tests\n"
	@printf "  lint                        Run ruff checks\n"
	@printf "  format                      Format Python files with ruff\n"
	@printf "  check                       Run tests and lint\n"
	@printf "  train-linear                Run linear-Gaussian training\n"
	@printf "  evaluate-linear             Run linear-Gaussian evaluation\n"
	@printf "  plot-linear                 Plot linear-Gaussian results\n"
	@printf "  train-nonlinear             Run nonlinear training\n"
	@printf "  evaluate-nonlinear          Run nonlinear evaluation\n"
	@printf "  sweep-nonlinear-reference   Run nonlinear grid reference suite\n"
	@printf "  sweep-nonlinear-learned     Run learned nonlinear filter suite\n"
	@printf "  plot-nonlinear              Plot nonlinear run diagnostics\n"
	@printf "  plot-nonlinear-sweep        Plot nonlinear sweep comparison metrics\n"
	@printf "  clean                       Remove local caches\n"

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
	$(UV) run python scripts/train_nonlinear.py --config $(TRAIN_NONLINEAR_CONFIG)

evaluate-nonlinear:
	$(UV) run python scripts/evaluate_nonlinear.py --config $(NONLINEAR_CONFIG)

sweep-nonlinear-reference:
	$(UV) run python scripts/sweep_nonlinear_reference.py --configs $(NONLINEAR_REFERENCE_CONFIGS) --output-dir $(NONLINEAR_REFERENCE_DIR)

sweep-nonlinear-learned:
	$(UV) run python scripts/sweep_nonlinear_learned.py --configs $(NONLINEAR_LEARNED_CONFIGS) --models $(NONLINEAR_LEARNED_MODELS) $(if $(NONLINEAR_LEARNED_SEEDS),--seeds $(NONLINEAR_LEARNED_SEEDS),) $(if $(NONLINEAR_REFERENCE_VARIANCE_RATIO_WEIGHT),--reference-variance-ratio-weight $(NONLINEAR_REFERENCE_VARIANCE_RATIO_WEIGHT),) $(if $(NONLINEAR_REFERENCE_TIME_VARIANCE_RATIO_WEIGHT),--reference-time-variance-ratio-weight $(NONLINEAR_REFERENCE_TIME_VARIANCE_RATIO_WEIGHT),) $(if $(NONLINEAR_REFERENCE_LOW_OBSERVATION_VARIANCE_RATIO_WEIGHT),--reference-low-observation-variance-ratio-weight $(NONLINEAR_REFERENCE_LOW_OBSERVATION_VARIANCE_RATIO_WEIGHT),) --steps $(NONLINEAR_LEARNED_STEPS) --output-dir $(NONLINEAR_LEARNED_DIR)

plot-nonlinear:
	$(UV) run python scripts/plot_nonlinear.py --run-dir $(RUN_DIR)

plot-nonlinear-sweep:
	$(UV) run python scripts/plot_nonlinear_sweep.py --metrics $(NONLINEAR_SWEEP_METRICS) --baseline-metrics $(NONLINEAR_SWEEP_BASELINE_METRICS) --weights $(NONLINEAR_SWEEP_WEIGHTS) --patterns $(NONLINEAR_SWEEP_PATTERNS) --output-dir $(NONLINEAR_SWEEP_PLOT_DIR)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
