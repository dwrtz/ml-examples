UV ?= uv

TRAIN_LINEAR_CONFIG ?= experiments/linear_gaussian/01_supervised_edge_mlp.yaml
TRAIN_LINEAR_ELBO_CONFIG ?= experiments/linear_gaussian/02_elbo_edge_mlp.yaml
EVALUATE_LINEAR_CONFIG ?= experiments/linear_gaussian/00_oracle_check.yaml
NONLINEAR_CONFIG ?= experiments/nonlinear/01_sine_observation.yaml
TRAIN_NONLINEAR_CONFIG ?= experiments/nonlinear/08_direct_elbo_sine_mlp.yaml
NONLINEAR_REFERENCE_DIR ?= outputs/nonlinear_reference_suite
NONLINEAR_REFERENCE_CONFIGS ?= experiments/nonlinear/01_sine_observation.yaml,experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml,experiments/nonlinear/05_zero_sine_observation.yaml,experiments/nonlinear/06_random_normal_sine_observation.yaml
NONLINEAR_LEARNED_DIR ?= outputs/nonlinear_learned_suite
NONLINEAR_LEARNED_CONFIGS ?= $(NONLINEAR_REFERENCE_CONFIGS)
NONLINEAR_LEARNED_MODELS ?= direct_elbo,structured_elbo
NONLINEAR_LEARNED_STEPS ?= 250
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
DIAGNOSTIC_BASELINES_DIR ?= outputs/linear_gaussian_diagnostic_baselines
DIAGNOSTIC_BASELINES_STEPS ?= 250,1000,3000
OBJECTIVE_BUDGET_DIR ?= outputs/linear_gaussian_objective_budget
OBJECTIVE_BUDGET_STEPS ?= 250,1000,3000
PREDICTIVE_HEAD_CONFIG ?= experiments/linear_gaussian/06_predictive_head.yaml
PREDICTIVE_HEAD_DIR ?= outputs/linear_gaussian_predictive_head
PREDICTIVE_HEAD_SWEEP_DIR ?= outputs/linear_gaussian_predictive_head_sweep
SELF_FED_SWEEP_DIR ?= outputs/linear_gaussian_self_fed_supervised
SELF_FED_VARIANCE_DIR ?= outputs/linear_gaussian_self_fed_variance_regularizer
SELF_FED_VARIANCE_WEIGHTS ?= 0,0.1,1,10
SELF_FED_VARIANCE_STEPS ?= 3000
WEAK_OBSERVABILITY_DIR ?= outputs/linear_gaussian_weak_observability
WEAK_OBSERVABILITY_MODELS ?= zero,frozen,self_fed,self_fed_calibrated,elbo,elbo_calibrated,direct_closed_form
WEAK_OBSERVABILITY_STEPS ?= 3000
ELBO_CALIBRATION_DIR ?= outputs/linear_gaussian_elbo_calibration
ELBO_CALIBRATION_PATTERNS ?= weak_sinusoidal,intermittent_sinusoidal,zero_unobservable
ELBO_CALIBRATION_WEIGHTS ?= 0,0.1,1
ELBO_CALIBRATION_PENALTIES ?= time,low_observation
WEAK_OBSERVABILITY_CANONICAL_DIR ?= outputs/linear_gaussian_weak_observability_canonical
LINEAR_GAUSSIAN_FINAL_REPORT_DIR ?= outputs/linear_gaussian_final_report
QR_GENERALIZATION_DIR ?= outputs/linear_gaussian_qr_generalization
QR_GENERALIZATION_MODELS ?= frozen,self_fed_calibrated,elbo_calibrated
QR_GENERALIZATION_TRAIN_PAIRS ?= 0.1:0.1
QR_GENERALIZATION_EVAL_PAIRS ?= 0.03:0.03,0.1:0.1,0.3:0.3,0.03:0.3,0.3:0.03
QR_GENERALIZATION_STEPS ?= 1000
RANDOM_QR_GENERALIZATION_DIR ?= outputs/linear_gaussian_random_qr_generalization
RANDOM_QR_GENERALIZATION_MODELS ?= self_fed_calibrated,elbo_calibrated
RANDOM_QR_GENERALIZATION_TRAIN_Q_VALUES ?= 0.03,0.1,0.3
RANDOM_QR_GENERALIZATION_TRAIN_R_VALUES ?= 0.03,0.1,0.3
RANDOM_QR_CALIBRATION_DIR ?= outputs/linear_gaussian_random_qr_calibration
RANDOM_QR_CALIBRATION_WEIGHTS ?= 0,0.1,1
RANDOM_QR_CANONICAL_DIR ?= outputs/linear_gaussian_random_qr_generalization_canonical

.PHONY: help setup lock test lint format check train-linear train-linear-elbo evaluate-linear plot-linear plot-linear-elbo plot-nonlinear compare-linear sweep-linear sweep-elbo-ablation sweep-edge-regularizer sweep-transition-consistency sweep-diagnostic-baselines sweep-objective-budget train-predictive-head sweep-predictive-head sweep-self-fed-supervised sweep-self-fed-variance sweep-weak-observability sweep-elbo-calibration aggregate-weak-observability aggregate-linear-gaussian-final-report aggregate-linear-gaussian-reports sweep-qr-generalization sweep-random-qr-generalization sweep-random-qr-calibration aggregate-random-qr-generalization train-nonlinear evaluate-nonlinear sweep-nonlinear-reference sweep-nonlinear-learned clean

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
	@printf "  plot-nonlinear     Plot nonlinear run diagnostics\n"
	@printf "  compare-linear     Compare supervised and ELBO linear-Gaussian runs\n"
	@printf "  sweep-linear       Train and aggregate linear-Gaussian seed sweep\n"
	@printf "  sweep-elbo-ablation Run ELBO MC-sample/training-budget ablation\n"
	@printf "  sweep-edge-regularizer Run diagnostic oracle edge-KL regularizer sweep\n"
	@printf "  sweep-transition-consistency Run unsupervised transition regularizer sweep\n"
	@printf "  sweep-diagnostic-baselines Run zero/frozen/split-head diagnostic baselines\n"
	@printf "  sweep-objective-budget Run matched-budget supervised-vs-ELBO sweep\n"
	@printf "  train-predictive-head Train one-step predictive head\n"
	@printf "  sweep-predictive-head Run predictive-head seed sweep\n"
	@printf "  sweep-self-fed-supervised Run teacher-forced/self-fed/ELBO sweep\n"
	@printf "  sweep-self-fed-variance Run self-fed variance-ratio regularizer sweep\n"
	@printf "  sweep-weak-observability Run weak-observability linear-Gaussian suite\n"
	@printf "  sweep-elbo-calibration Run targeted ELBO calibration sweep\n"
	@printf "  aggregate-weak-observability Merge split weak-observability summaries\n"
	@printf "  aggregate-linear-gaussian-final-report Build scalar Gaussian final report\n"
	@printf "  aggregate-linear-gaussian-reports Rebuild all scalar Gaussian reports\n"
	@printf "  sweep-qr-generalization Run fixed-Q/R generalization suite\n"
	@printf "  sweep-random-qr-generalization Run randomized-Q/R generalization suite\n"
	@printf "  sweep-random-qr-calibration Run randomized-Q/R calibration sweep\n"
	@printf "  aggregate-random-qr-generalization Merge randomized-Q/R canonical summaries\n"
	@printf "  train-nonlinear    Run nonlinear training\n"
	@printf "  evaluate-nonlinear Run nonlinear evaluation\n"
	@printf "  sweep-nonlinear-reference Run nonlinear grid reference suite\n"
	@printf "  sweep-nonlinear-learned Run learned nonlinear filter suite\n"
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

plot-nonlinear:
	$(UV) run python scripts/plot_nonlinear.py --run-dir $(RUN_DIR)

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

sweep-diagnostic-baselines:
	$(UV) run python scripts/sweep_diagnostic_baselines.py --seeds $(LINEAR_SWEEP_SEEDS) --steps $(DIAGNOSTIC_BASELINES_STEPS) --output-dir $(DIAGNOSTIC_BASELINES_DIR)

sweep-objective-budget:
	$(UV) run python scripts/sweep_objective_budget.py --seeds $(LINEAR_SWEEP_SEEDS) --steps $(OBJECTIVE_BUDGET_STEPS) --output-dir $(OBJECTIVE_BUDGET_DIR)

train-predictive-head:
	$(UV) run python scripts/train_predictive_head.py --config $(PREDICTIVE_HEAD_CONFIG)

sweep-predictive-head:
	$(UV) run python scripts/sweep_predictive_head.py --seeds $(LINEAR_SWEEP_SEEDS) --output-dir $(PREDICTIVE_HEAD_SWEEP_DIR)

sweep-self-fed-supervised:
	$(UV) run python scripts/sweep_self_fed_supervised.py --seeds $(LINEAR_SWEEP_SEEDS) --steps $(OBJECTIVE_BUDGET_STEPS) --output-dir $(SELF_FED_SWEEP_DIR)

sweep-self-fed-variance:
	$(UV) run python scripts/sweep_self_fed_variance_regularizer.py --seeds $(LINEAR_SWEEP_SEEDS) --weights $(SELF_FED_VARIANCE_WEIGHTS) --steps $(SELF_FED_VARIANCE_STEPS) --output-dir $(SELF_FED_VARIANCE_DIR)

sweep-weak-observability:
	$(UV) run python scripts/sweep_weak_observability.py --seeds $(LINEAR_SWEEP_SEEDS) --models $(WEAK_OBSERVABILITY_MODELS) --steps $(WEAK_OBSERVABILITY_STEPS) --output-dir $(WEAK_OBSERVABILITY_DIR)

sweep-elbo-calibration:
	$(UV) run python scripts/sweep_elbo_calibration.py --seeds $(LINEAR_SWEEP_SEEDS) --patterns $(ELBO_CALIBRATION_PATTERNS) --weights $(ELBO_CALIBRATION_WEIGHTS) --penalties $(ELBO_CALIBRATION_PENALTIES) --steps $(WEAK_OBSERVABILITY_STEPS) --output-dir $(ELBO_CALIBRATION_DIR)

aggregate-weak-observability:
	$(UV) run python scripts/aggregate_weak_observability.py --output-dir $(WEAK_OBSERVABILITY_CANONICAL_DIR)

aggregate-linear-gaussian-final-report:
	$(UV) run python scripts/aggregate_linear_gaussian_final_report.py --output-dir $(LINEAR_GAUSSIAN_FINAL_REPORT_DIR)

aggregate-linear-gaussian-reports: aggregate-weak-observability aggregate-random-qr-generalization aggregate-linear-gaussian-final-report

sweep-qr-generalization:
	$(UV) run python scripts/sweep_qr_generalization.py --seeds $(LINEAR_SWEEP_SEEDS) --models $(QR_GENERALIZATION_MODELS) --steps $(QR_GENERALIZATION_STEPS) --train-pairs $(QR_GENERALIZATION_TRAIN_PAIRS) --eval-pairs $(QR_GENERALIZATION_EVAL_PAIRS) --output-dir $(QR_GENERALIZATION_DIR)

sweep-random-qr-generalization:
	$(UV) run python scripts/sweep_random_qr_generalization.py --seeds $(LINEAR_SWEEP_SEEDS) --models $(RANDOM_QR_GENERALIZATION_MODELS) --steps $(QR_GENERALIZATION_STEPS) --train-q-values $(RANDOM_QR_GENERALIZATION_TRAIN_Q_VALUES) --train-r-values $(RANDOM_QR_GENERALIZATION_TRAIN_R_VALUES) --eval-pairs $(QR_GENERALIZATION_EVAL_PAIRS) --output-dir $(RANDOM_QR_GENERALIZATION_DIR)

sweep-random-qr-calibration:
	$(UV) run python scripts/sweep_random_qr_calibration.py --seeds $(LINEAR_SWEEP_SEEDS) --weights $(RANDOM_QR_CALIBRATION_WEIGHTS) --steps $(QR_GENERALIZATION_STEPS) --train-q-values $(RANDOM_QR_GENERALIZATION_TRAIN_Q_VALUES) --train-r-values $(RANDOM_QR_GENERALIZATION_TRAIN_R_VALUES) --eval-pairs $(QR_GENERALIZATION_EVAL_PAIRS) --output-dir $(RANDOM_QR_CALIBRATION_DIR)

aggregate-random-qr-generalization:
	$(UV) run python scripts/aggregate_random_qr_generalization.py --output-dir $(RANDOM_QR_CANONICAL_DIR)

train-nonlinear:
	$(UV) run python scripts/train_nonlinear.py --config $(TRAIN_NONLINEAR_CONFIG)

evaluate-nonlinear:
	$(UV) run python scripts/evaluate_nonlinear.py --config $(NONLINEAR_CONFIG)

sweep-nonlinear-reference:
	$(UV) run python scripts/sweep_nonlinear_reference.py --configs $(NONLINEAR_REFERENCE_CONFIGS) --output-dir $(NONLINEAR_REFERENCE_DIR)

sweep-nonlinear-learned:
	$(UV) run python scripts/sweep_nonlinear_learned.py --configs $(NONLINEAR_LEARNED_CONFIGS) --models $(NONLINEAR_LEARNED_MODELS) --steps $(NONLINEAR_LEARNED_STEPS) --output-dir $(NONLINEAR_LEARNED_DIR)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
