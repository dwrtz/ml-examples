# Variational Bayesian Filtering Modernization: Research Progress Note

Prepared: 2026-04-24

## 1. Executive summary

We have modernized the original learned Kalman-filtering experiment into a small, testable benchmark for learned variational Bayesian filtering. The current implementation preserves the central research object:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

The scalar linear-Gaussian benchmark is now running with:

- exact Kalman filtering;
- exact two-state edge posterior oracle;
- supervised edge-posterior distillation;
- unsupervised edge-local ELBO training;
- held-out evaluation;
- multi-seed sweeps;
- ELBO term diagnostics;
- learned-vs-oracle ELBO term references;
- one-step posterior predictive evaluation.

The most important empirical result so far is that the unsupervised edge ELBO, when trained with enough Monte Carlo samples and optimization steps, recovers a useful filtering state and outperforms the current supervised strict-MLP baseline on learned filter KL, edge KL, state RMSE, state NLL, and one-step predictive NLL. Exact Kalman remains the calibration reference and is still substantially better on state NLL and predictive NLL.

## 2. Current implementation state

The active benchmark is the scalar dynamic regression model:

```text
z_t = z_{t-1} + w_t,       w_t ~ Normal(0, Q)
y_t = x_t z_t + v_t,       v_t ~ Normal(0, R)
z_0 ~ Normal(m_0, P_0)
```

Implemented components:

| Component | Status |
|---|---|
| Linear-Gaussian data generator | implemented |
| Exact scalar Kalman filter | implemented |
| Exact two-state edge posterior oracle | implemented |
| Gaussian filtering belief and backward conditional | implemented |
| Strict structured MLP update cell | implemented |
| Supervised edge KL objective | implemented |
| Unsupervised edge-local ELBO objective | implemented |
| Held-out evaluation batches | implemented |
| Standard train/evaluate/plot workflow | implemented |
| Multi-seed comparison sweep | implemented |
| ELBO MC-sample/training-budget ablation | implemented |
| Learned and oracle ELBO term diagnostics | implemented |
| One-step predictive metrics from `q^F_{t-1}` | implemented |

The default ELBO config is now:

```yaml
training:
  steps: 1000
  num_elbo_samples: 32
  edge_kl_weight: 0.0
  transition_consistency_weight: 0.0
```

The nonzero regularizers are available for diagnostic sweeps but are not part of the default unsupervised objective.

## 3. Main quantitative results

All results below use held-out evaluation episodes. The current standard seed sweep uses five seeds: `321, 322, 323, 324, 325`.

### 3.1 Five-seed standard comparison

| Model | Objective | filter KL | edge KL | state RMSE | state NLL | pred NLL |
|---|---|---:|---:|---:|---:|---:|
| MLP edge ELBO | unsupervised ELBO | 0.139600 +/- 0.020257 | 0.416881 +/- 0.043102 | 0.436199 +/- 0.003547 | 0.542554 +/- 0.023419 | 0.640331 +/- 0.006809 |
| MLP supervised edge KL | oracle distillation | 0.228665 +/- 0.010500 | 0.449262 +/- 0.010241 | 0.502511 +/- 0.002307 | 0.629996 +/- 0.007986 | 0.708275 +/- 0.007953 |
| exact Kalman | oracle | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.522426 +/- 0.003409 | 0.401983 +/- 0.003410 | 0.600858 +/- 0.003707 |

Interpretation:

- ELBO training now beats the current supervised strict-MLP baseline on all learned-model metrics.
- Exact Kalman remains much better calibrated by state NLL and predictive NLL.
- Learned models can produce lower point RMSE than Kalman on these finite held-out samples, but Kalman remains the probabilistic reference.
- Predictive NLL confirms that the ELBO-trained carried filtering state `q^F_{t-1}` is meaningfully better for one-step prediction than the supervised baseline.

### 3.2 ELBO training budget and sample-count ablation

The ELBO ablation varied MC samples and training steps across the same five seeds.

| Steps | MC samples | filter KL | edge KL | state RMSE | state NLL |
|---:|---:|---:|---:|---:|---:|
| 250 | 8 | 0.195375 | 0.753137 | 0.448535 | 0.598567 |
| 250 | 32 | 0.189909 | 0.726516 | 0.447866 | 0.593159 |
| 1000 | 8 | 0.147975 | 0.437122 | 0.438186 | 0.550803 |
| 1000 | 32 | 0.139600 | 0.416881 | 0.436199 | 0.542554 |

Interpretation:

- Training budget is the primary lever.
- More MC samples help steadily, but moving from 250 to 1000 steps closes most of the edge KL gap.
- The promoted default `1000` steps and `32` MC samples is justified by this ablation.

### 3.3 ELBO term diagnostics

For the promoted ELBO run on seed `321`, learned-vs-oracle ELBO term means are:

| Term | Learned | Oracle |
|---|---:|---:|
| ELBO | -0.771693 | -0.595622 |
| log likelihood | -0.328621 | -0.265193 |
| log transition | -0.385376 | -0.265594 |
| log previous filter | -0.239154 | -0.422730 |
| negative log current filter | 0.272691 | 0.403733 |
| negative log backward | -0.091232 | -0.045837 |

Interpretation:

- The remaining learned-vs-oracle ELBO gap is mainly in observation fit and transition consistency.
- The learned model partially compensates through previous-filter/current-filter terms.
- This suggests the failure mode is not simply the filtering marginal; it is tied to edge dynamics and the backward conditional.

## 4. Diagnostic regularizer experiments

### 4.1 Oracle edge-KL regularizer

We added a diagnostic linear-Gaussian regularizer:

```text
loss = -ELBO + lambda * KL(q_oracle_edge || q_learned_edge)
```

This is not a candidate final unsupervised objective because it uses the oracle. It is a capacity and optimization diagnostic.

Five-seed sweep:

| edge KL weight | filter KL | edge KL | state RMSE | state NLL | ELBO |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.139600 | 0.416881 | 0.436199 | 0.542554 | -0.782543 |
| 0.01 | 0.126073 | 0.374292 | 0.435351 | 0.528946 | -0.779742 |
| 0.05 | 0.097496 | 0.298454 | 0.433380 | 0.500040 | -0.775072 |
| 0.1 | 0.080787 | 0.250151 | 0.432124 | 0.483183 | -0.770316 |

Interpretation:

- The posterior family and MLP update have capacity to represent a substantially better edge posterior.
- The regularizer improves not only edge KL but also filter KL, state NLL, point RMSE, and evaluated ELBO.
- This points toward an optimization/objective guidance issue, not a hard architectural expressivity limit.

### 4.2 Naive transition-consistency regularizer

We tested an unsupervised transition residual moment penalty that encourages:

```text
z_t - z_{t-1}
```

to have mean near zero and second moment near `Q` under the learned edge posterior.

Five-seed sweep:

| transition weight | filter KL | edge KL | state RMSE | state NLL | ELBO |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.139600 | 0.416881 | 0.436199 | 0.542554 | -0.782543 |
| 0.01 | 0.164154 | 0.473065 | 0.440802 | 0.567770 | -0.798680 |
| 0.05 | 0.285967 | 0.719218 | 0.458435 | 0.690890 | -0.855190 |
| 0.1 | 0.388996 | 0.895882 | 0.473668 | 0.794520 | -0.897804 |

Interpretation:

- This penalty hurts monotonically.
- The mistake is likely conceptual: after conditioning on `y_t`, the edge residual should not simply match the unconditional transition noise moment.
- This negative result is useful because it rules out a tempting but too-blunt unsupervised regularizer.

## 5. Predictive evaluation

We added a model-consistent one-step predictive distribution from the carried filtering marginal:

```text
z_{t-1} ~ q^F_{t-1}
z_t     ~ Normal(z_{t-1}, Q)
y_t     ~ Normal(x_t z_t, R)
```

Closed-form moments:

```text
mean_y = x_t * previous_filter_mean
var_y  = x_t^2 * (previous_filter_var + Q) + R
```

This metric checks whether `q^F_t` is actually a useful filtering state, not merely close after assimilation.

Current five-seed predictive NLL:

| Model | pred NLL |
|---|---:|
| exact Kalman | 0.600858 +/- 0.003707 |
| MLP edge ELBO | 0.640331 +/- 0.006809 |
| MLP supervised edge KL | 0.708275 +/- 0.007953 |

Interpretation:

- ELBO produces a much better carried predictive state than the supervised baseline.
- Kalman remains the predictive calibration target.
- This supports moving next to an explicit learned predictive head, while keeping the analytic model-consistent predictive as the baseline.

## 6. Current research interpretation

The current results support several conclusions:

1. The two-state edge posterior formulation is working as a research object.
2. A local unsupervised ELBO can recover useful filtering behavior in the linear-Gaussian case.
3. The default ELBO setting needs enough optimization budget and MC samples; the initial small setting understated ELBO performance.
4. The strict MLP architecture has enough capacity to improve substantially when given oracle edge guidance.
5. The remaining unsupervised gap is not solved by naive transition residual moment matching.
6. Predictive evaluation is now in place and shows the ELBO-trained filter state is meaningfully useful before assimilation.

The strongest open issue is calibration. Exact Kalman remains much better on state NLL and predictive NLL, even when learned point RMSE is lower.

## 7. Recommended next steps

### 7.1 Near-term engineering/research milestone

Add a learned predictive head:

```text
r_omega(y_t | q^F_{t-1}, x_t)
```

Requirements:

- input must be previous filter mean/variance, `x_t`, and known model features such as `Q/R`;
- current `y_t` must only appear as the training target;
- include no-leakage tests;
- compare learned predictive NLL against exact Kalman predictive NLL and analytic model-consistent predictive NLL.

Why this should be next:

- It is already listed in the handoff plan.
- Predictive metrics are now available.
- It tests whether learned filters produce reusable Bayesian state summaries.

### 7.2 Objective research direction

Investigate unsupervised edge guidance that is less blunt than residual moment matching. Candidate directions:

- use the exact transition likelihood term more carefully, possibly with lower-variance analytic expectations where available;
- add control variates or antithetic samples for ELBO term variance reduction;
- compare Monte Carlo ELBO to closed-form local ELBO in the scalar Gaussian case as a debugging baseline;
- regularize the backward conditional through terms derived from the local model structure rather than unconditional transition moments.

### 7.3 Architecture milestone

After predictive-head evaluation, add recurrent/context-assisted amortization:

- GRU/LSTM first, for historical comparison;
- Mamba/Mamba-2 only after the strict-filter baseline and predictive-head metrics are stable.

The key reporting requirement should remain whether predictions can be made from `q^F_t` alone. Otherwise recurrent context may become hidden posterior state and obscure the filtering interpretation.

## 8. Questions for research-director feedback

1. Should the next priority be the learned predictive head, or a deeper objective-variance study of the ELBO?
2. Is the current strict-MLP baseline sufficient for the first paper-style comparison, or should GRU/LSTM be added before nonlinear experiments?
3. How should we treat lower learned point RMSE but worse NLL than Kalman in reporting? Current interpretation is that Kalman remains the calibration reference.
4. Are diagnostic oracle-regularized results useful to include as an upper-bound/capacity study, or should they remain internal?
5. Should nonlinear experiments start after the predictive head, or should the linear-Gaussian benchmark first include closed-form ELBO baselines and stronger calibration analysis?

## 9. Reproducibility commands

Standard comparison:

```bash
make train-linear
make train-linear-elbo
make plot-linear
make plot-linear-elbo
make compare-linear
```

Five-seed learned-model comparison:

```bash
make sweep-linear
```

ELBO MC-sample/training-budget ablation:

```bash
make sweep-elbo-ablation
```

Diagnostic oracle edge regularizer:

```bash
make sweep-edge-regularizer
```

Naive unsupervised transition-consistency regularizer:

```bash
make sweep-transition-consistency
```

All generated outputs are ignored under `outputs/`; the source code, configs, and scripts are committed.
