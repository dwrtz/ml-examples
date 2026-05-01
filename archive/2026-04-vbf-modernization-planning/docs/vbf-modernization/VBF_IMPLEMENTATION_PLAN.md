# Variational Bayesian Filtering Modernization: Implementation Plan

Prepared: 2026-04-24

## 1. Purpose

This plan modernizes the original “can you learn to do Kalman filtering?” experiment into a small research benchmark for **learned variational Bayesian filtering**.

The core idea to preserve is not “replace a Kalman filter with an RNN.” The core idea is:

1. At each time step, form a **local two-state filtering problem** over `(z_t, z_{t-1})`.
2. Fit a **variational edge posterior** over the two consecutive states.
3. Carry forward a **filtering marginal** over `z_t`.
4. Train with a local sequential variational objective.
5. Compare against exact Kalman filtering in the linear-Gaussian case, then move to harder cases.

The original code already implements the important conceptual pieces: a dynamic scalar regression data generator, an exact Kalman baseline, and a learned variational posterior over `[z_t, z_{t-1}]`. The modernization should make those ideas explicit, testable, reusable, and extensible.

## 2. Non-negotiable design requirements

### 2.1 Preserve the two-state edge posterior

At time `t`, the local joint should be treated as:

```text
p(y_t, z_t, z_{t-1} | D_{1:t-1}, x_t)
  = p(y_t | z_t, x_t) p(z_t | z_{t-1}) q^F_{t-1}(z_{t-1})
```

where:

```text
D_{1:t} = {(x_1, y_1), ..., (x_t, y_t)}
q^F_{t-1}(z_{t-1}) ≈ p(z_{t-1} | D_{1:t-1})
```

The variational posterior at time `t` must be over the edge:

```text
q^E_t(z_t, z_{t-1}) ≈ p(z_t, z_{t-1} | D_{1:t})
```

This is the crucial point from the original experiment. A one-state posterior over `z_t` alone is not enough unless the predictive prior has already been analytically collapsed. In the general case, that collapse is the hard part.

### 2.2 Make the filtering marginal explicit

In the general case, marginalizing `z_{t-1}` out of an arbitrary learned joint may be analytically intractable. Therefore, the recommended posterior parameterization is:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

This makes the filtering posterior explicit by construction:

```text
∫ q^E_t(z_t, z_{t-1}) dz_{t-1} = q^F_t(z_t)
```

`q^F_t` is the belief that gets carried forward to the next time step.

### 2.3 Keep the ELBO edge-local

The local ELBO is:

```text
L_t = E_{q^E_t}[
    log p(y_t | z_t, x_t)
  + log p(z_t | z_{t-1})
  + log q^F_{t-1}(z_{t-1})
  - log q^F_t(z_t)
  - log q^B_t(z_{t-1} | z_t)
]
```

Training should maximize `sum_t L_t`, or minimize `-sum_t L_t`.

### 2.4 Separate strict filtering from context-assisted amortization

There are two legitimate modes:

**Strict learned filter mode**

```text
q^F_t = update(q^F_{t-1}, x_t, y_t)
```

No other hidden state is carried. This is the cleanest test of whether the learned object is actually a filter.

**Context-assisted amortization mode**

```text
c_t   = context_model(c_{t-1}, q^F_{t-1}, x_t, y_t)
q^F_t = update(q^F_{t-1}, x_t, y_t, c_t)
```

This permits GRU, LSTM, Mamba, Mamba-2, Transformer-XL-like recurrence, or other sequence backbones. However, experiments must report whether downstream predictions can be made from `q^F_t` alone. Otherwise the context model can become a hidden posterior state and obscure the Bayesian filtering interpretation.

## 3. Baseline state-space model

Start with the same scalar dynamic regression model used by the original experiment:

```text
z_t = z_{t-1} + w_t,       w_t ~ Normal(0, Q)
y_t = x_t z_t + v_t,       v_t ~ Normal(0, R)
z_0 ~ Normal(m_0, P_0)
```

`x_t` is observed and may be deterministic, sinusoidal, random, missing, or adversarially uninformative.

The first benchmark should use known `Q`, `R`, `m_0`, and `P_0`. After the inference architecture is validated, add experiments where `Q`, `R`, or parts of the transition/observation models are learned.

## 4. Exact Kalman and edge-posterior oracle

The linear-Gaussian model has an exact solution. Implement an oracle that returns both:

1. the usual filtering marginal `p(z_t | D_{1:t})`; and
2. the exact two-state edge posterior `p(z_t, z_{t-1} | D_{1:t})`.

This oracle is essential for debugging, supervised distillation, and evaluation.

### 4.1 Standard scalar Kalman recursion

Given previous filtering posterior:

```text
z_{t-1} | D_{1:t-1} ~ Normal(m_{t-1}, P_{t-1})
```

Prediction:

```text
m_pred = m_{t-1}
P_pred = P_{t-1} + Q
```

Innovation:

```text
e_t = y_t - x_t m_pred
S_t = x_t^2 P_pred + R
K_t = P_pred x_t / S_t
```

Update:

```text
m_t = m_pred + K_t e_t
P_t = (1 - K_t x_t) P_pred
```

### 4.2 Exact two-state edge posterior

Define the edge vector:

```text
u_t = [z_t, z_{t-1}]^T
```

Before observing `y_t`, the prior over the edge is:

```text
mu_edge_prior = [m_{t-1}, m_{t-1}]^T

Sigma_edge_prior = [[P_{t-1} + Q, P_{t-1}],
                    [P_{t-1},     P_{t-1}]]
```

The measurement model is:

```text
y_t = [x_t, 0] u_t + v_t,   v_t ~ Normal(0, R)
```

Let:

```text
H_t = [x_t, 0]
S_t = H_t Sigma_edge_prior H_t^T + R
K_edge_t = Sigma_edge_prior H_t^T / S_t
```

Then:

```text
mu_edge_t = mu_edge_prior + K_edge_t (y_t - H_t mu_edge_prior)
Sigma_edge_t = Sigma_edge_prior - K_edge_t S_t K_edge_t^T
```

The filtering marginal is recovered from the first component:

```text
m_t = mu_edge_t[0]
P_t = Sigma_edge_t[0, 0]
```

This oracle should be implemented in `src/vbf/kalman.py` and unit-tested against the standard scalar Kalman recursion.

## 5. Recommended posterior family for Experiment 0

Use a scalar Gaussian forward belief and a scalar Gaussian backward conditional:

```text
q^F_t(z_t) = Normal(mu_t, sigma_t^2)

q^B_t(z_{t-1} | z_t) = Normal(a_t z_t + b_t, tau_t^2)
```

Then:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

This posterior family can represent any non-degenerate bivariate Gaussian over `(z_t, z_{t-1})`. If the target edge posterior has mean:

```text
E[z_t] = m_t
E[z_{t-1}] = m_prev_smooth
```

and covariance:

```text
Var[z_t] = V_t
Var[z_{t-1}] = V_prev_smooth
Cov[z_t, z_{t-1}] = C_t
```

then the conditional parameters are:

```text
mu_t = m_t
sigma_t^2 = V_t
a_t = C_t / V_t
b_t = m_prev_smooth - a_t m_t
tau_t^2 = V_prev_smooth - C_t^2 / V_t
```

This makes it an ideal first family: it preserves the two-state edge posterior, supports exact log probability and reparameterized sampling, and exposes the filtering marginal directly.

Implementation notes:

- Parameterize all variances with `softplus(raw) + min_scale`.
- Use `log_variance` or unconstrained scale parameters internally.
- Clamp or regularize extreme log-scales for numerical stability.
- Prefer `float64` for the Kalman oracle and closed-form test calculations; model training can use `float32` unless instability appears.

## 6. Core implementation modules

Recommended package namespace: `vbf`.

### 6.1 `src/vbf/data.py`

Responsibilities:

- Generate batched episodes.
- Support fixed and randomized noise regimes.
- Support multiple `x_t` patterns.
- Return true latent states for synthetic evaluation.

Suggested episode fields:

```text
x:        [batch, time, x_dim]
y:        [batch, time, y_dim]
z:        [batch, time, z_dim]
Q:        [batch] or scalar
R:        [batch] or scalar
m0:       [batch] or scalar
P0:       [batch] or scalar
metadata: dict
```

Initial data regimes:

| Regime | Description | Purpose |
|---|---|---|
| `sinusoid_fixed_qr` | Original-style sinusoidal `x_t`, fixed `Q/R` | Reproduce original experiment |
| `sinusoid_random_qr` | Randomize `Q` and `R` per episode | Test whether the learned filter generalizes |
| `near_zero_x` | Long stretches with `x_t ≈ 0` | Test uncertainty growth under weak observability |
| `random_x` | Independent or AR covariate sequence | Avoid overfitting to sinusoid pattern |
| `long_sequence` | Longer `T` than training | Test recurrence stability |
| `missing_y` | Mask some observations | Prepare for real filtering tasks |

### 6.2 `src/vbf/kalman.py`

Responsibilities:

- Exact scalar Kalman filtering.
- Exact scalar two-state edge posterior.
- Exact one-step measurement predictive distribution.
- Optional batched vectorized implementation.

Outputs per time step:

```text
filter_mean:         m_t
filter_var:          P_t
pred_mean:           m_pred
pred_var:            P_pred
innovation:          e_t
innovation_var:      S_t
kalman_gain:         K_t
edge_mean:           [E z_t, E z_{t-1}]
edge_cov:            2x2 covariance
measurement_pred_mu: x_t m_pred
measurement_pred_var:x_t^2 P_pred + R
```

### 6.3 `src/vbf/distributions.py`

Responsibilities:

- Implement distribution objects used by the learned filter.
- Keep sampling and log-probability logic centralized.

Suggested classes:

```text
GaussianBelief
ConditionalGaussianBackward
GaussianEdgePosterior
```

`GaussianEdgePosterior` should expose:

```text
sample(num_samples, reparameterized=True)
log_prob(z_t, z_tm1)
filtering_belief()
edge_mean_cov()
entropy_estimate_or_exact()
```

For the scalar Gaussian conditional family, exact joint mean/covariance and exact entropy are available.

### 6.4 `src/vbf/models/cells.py`

Responsibilities:

- Parameterize the learned update from previous belief and current observation.

Initial cells:

| Cell | Purpose |
|---|---|
| `StructuredMLPCell` | First strict filtering baseline |
| `GRUContextCell` | Direct comparison to older LSTM-style approach |
| `LSTMContextCell` | Historical comparison |

The strict `StructuredMLPCell` should consume only:

```text
features(q^F_{t-1}), x_t, y_t, optional model params
```

and output parameters for:

```text
q^F_t(z_t)
q^B_t(z_{t-1} | z_t)
```

### 6.5 `src/vbf/models/mamba_context.py`

Responsibilities:

- Add optional Mamba/Mamba-2-style causal context.
- Keep the exported belief explicit.

Suggested interface:

```text
context_t = context_model(context_{t-1}, features(q^F_{t-1}), x_t, y_t)
posterior_params_t = update_head(features(q^F_{t-1}), x_t, y_t, context_t)
```

Important diagnostics:

- Report all metrics with the full context model.
- Also report metrics when downstream prediction uses `q^F_t` only.
- Add ablations where context is reset or detached to determine whether the learned belief itself is sufficient.

### 6.6 `src/vbf/losses.py`

Responsibilities:

Implement:

1. edge ELBO loss;
2. supervised filtering KL loss;
3. supervised edge KL loss;
4. state NLL loss for synthetic latents;
5. predictive NLL loss;
6. hybrid objective wrappers.

The edge ELBO should operate as:

```text
For each t:
    q_edge_t = model.update(q_filter_{t-1}, x_t, y_t)
    sample z_t, z_{t-1} ~ q_edge_t
    score = log p(y_t | z_t, x_t)
          + log p(z_t | z_{t-1})
          + log q_filter_{t-1}(z_{t-1})
          - log q_edge_t(z_t, z_{t-1})
    q_filter_t = q_edge_t.filtering_belief()
```

Use Monte Carlo estimates first. For the Gaussian baseline, also implement closed-form expectations where practical as a debugging aid, but do not rely on closed-form expectations for the research path.

### 6.7 `src/vbf/predictive.py`

Responsibilities:

Represent learned posterior predictive distributions for measurements.

There are three distinct predictive objects:

| Object | Conditions on | Meaning |
|---|---|---|
| One-step pre-assimilation predictive | `q^F_{t-1}`, `x_t` | Forecast `y_t` before seeing it |
| Current posterior predictive | `q^F_t`, `x_t` | Replicated current measurement after assimilation |
| Future predictive | `q^F_t`, `x_{t+1}` | Forecast next measurement |

The most important one for evaluation is the **one-step pre-assimilation predictive**:

```text
r_omega(y_t | q^F_{t-1}, x_t) ≈ p(y_t | D_{1:t-1}, x_t)
```

In the scalar linear-Gaussian case, the exact predictive is:

```text
y_t | D_{1:t-1}, x_t ~ Normal(x_t m_pred, x_t^2 P_pred + R)
```

The learned predictive head must not condition on `y_t` when predicting `y_t`.

### 6.8 `src/vbf/metrics.py`

Responsibilities:

- Closed-form KL for 1D Gaussian marginals.
- Closed-form KL for 2D Gaussian edge posteriors.
- RMSE of posterior mean.
- NLL of true latent state under learned `q^F_t`.
- Calibration and interval coverage.
- Predictive NLL for `y_t`.
- Kalman gain diagnostics.
- Variance-growth diagnostics during uninformative measurements.

Recommended metrics:

| Metric | Description |
|---|---|
| `filter_kl` | KL between learned `q^F_t` and exact Kalman filtering marginal |
| `edge_kl` | KL between learned `q^E_t` and exact edge posterior |
| `state_rmse` | RMSE of learned posterior mean vs true latent `z_t` |
| `state_nll` | NLL of true latent `z_t` under learned `q^F_t` |
| `coverage_50/90/95` | Interval calibration |
| `prequential_y_nll` | NLL of `y_t` before assimilation |
| `gain_corr` | Correlation between learned update strength and Kalman gain |
| `weak_obs_var_slope` | Variance growth when `x_t ≈ 0` |

### 6.9 `src/vbf/train.py`

Responsibilities:

- Generic training loop.
- Checkpointing.
- Evaluation hooks.
- Logging.
- Seed control.
- Gradient clipping.
- NaN/Inf detection.

The training loop should support sequence scans and should not require fixed sequence length at model construction time.

### 6.10 `scripts/`

Suggested scripts:

```text
scripts/train_linear_gaussian.py
scripts/evaluate_linear_gaussian.py
scripts/plot_linear_gaussian.py
scripts/train_nonlinear.py
scripts/evaluate_nonlinear.py
```

Scripts should load YAML configs from `experiments/`.

## 7. Training ladder

### Stage A: oracle and data validation

Goal: prove the simulator and Kalman oracle are correct.

Tasks:

1. Generate episodes from the scalar linear-Gaussian model.
2. Run exact Kalman filtering.
3. Run exact edge posterior computation.
4. Check that the edge marginal over `z_t` equals the standard Kalman filtering posterior.
5. Check that the exact measurement predictive matches empirical held-out measurements.

Definition of done:

- Unit tests pass for edge marginal consistency.
- Kalman posterior tracks synthetic `z_t` with expected calibration.
- Plots reproduce the spirit of the original experiment.

### Stage B: supervised edge-posterior distillation

Goal: test whether the model family and architecture can represent the exact edge posterior.

Train the model to minimize:

```text
sum_t KL(p_oracle_edge_t || q_learned_edge_t)
```

or the reverse KL if that is easier to implement. Use both only if comparing mode-covering vs mode-seeking behavior becomes relevant.

Definition of done:

- `edge_kl` is near zero on train and same-distribution validation.
- `filter_kl` is near zero because the filtering marginal is explicit.
- Learned variances remain positive and calibrated.
- The conditional factor `q^B_t(z_{t-1} | z_t)` recovers the exact edge covariance.

### Stage C: unsupervised ELBO recovery

Goal: answer the original research question more directly.

Remove the oracle target and train only with the edge ELBO:

```text
maximize sum_t L_t
```

Evaluation still compares against the exact Kalman oracle, but the oracle is not used for training.

Definition of done:

- ELBO-trained model recovers the Kalman filtering marginal on same-distribution validation.
- Edge posterior is not just good in its `z_t` marginal; it also has the correct cross-time covariance.
- Uncertainty grows properly when `x_t` is uninformative.

### Stage D: posterior predictive measurement head

Goal: learn or approximate measurement predictive distributions.

Add:

```text
r_omega(y_t | q^F_{t-1}, x_t)
```

Train with one or both of:

```text
-log r_omega(y_t | q^F_{t-1}, x_t)
KL(exact_predictive_t || r_omega_t)       # only in linear-Gaussian oracle experiments
```

Definition of done:

- Predictive NLL is close to exact Kalman predictive NLL in the linear-Gaussian case.
- Predictive intervals for `y_t` are calibrated.
- The predictive head does not use post-assimilation information about `y_t`.

### Stage E: generalization suite

Goal: determine whether the learned filter is robust or merely memorized one regime.

Evaluate on:

| Suite | Description |
|---|---|
| Same distribution | Same generator as training |
| Longer sequences | `T_eval > T_train` |
| Different `Q/R` | Noise levels outside training range |
| Weak observability | Long spans with `x_t ≈ 0` |
| Different `x_t` family | Random or piecewise patterns instead of sinusoid |
| Missing measurements | Random masks on `y_t` |

Definition of done:

- The strict MLP cell should pass same-distribution and weak-observability tests before introducing large sequence backbones.
- Any Mamba/GRU/LSTM improvement should be reported separately for same-distribution and out-of-distribution settings.

### Stage F: architecture comparison

Goal: determine which architecture improves amortized inference without violating the filtering contract.

Compare:

1. strict structured MLP update cell;
2. GRU context cell;
3. LSTM context cell;
4. Mamba/Mamba-2 context cell;
5. optionally, small causal Transformer baseline.

Report:

- total parameters;
- training time;
- memory use;
- sequence length scaling;
- filtering metrics;
- predictive metrics;
- context ablation results.

### Stage G: nonlinear / non-Gaussian extensions

Only after the linear-Gaussian benchmark is working, move to harder models:

| Extension | Example |
|---|---|
| Nonlinear observation | `y_t = sin(z_t) + noise` or `y_t = h_theta(z_t, x_t) + noise` |
| Heavy-tailed observation noise | Student-t likelihood |
| Nonlinear transition | `z_t = f(z_{t-1}) + w_t` |
| State-dependent noise | `Q(z_{t-1})`, `R(z_t, x_t)` |
| Multidimensional state | Vector `z_t` with matrix-valued covariance |
| Unknown model parameters | Learn `Q`, `R`, or neural transition/observation models |

For these cases, the edge posterior should remain two-state. The filtering marginal should remain explicit, using conditional/factorized posteriors, flows, mixtures, or particles as needed.

## 8. Architecture recommendations

### 8.1 Start with a strict structured update cell

The first model should not use Mamba. It should use a small MLP that receives:

```text
mu_{t-1}, logvar_{t-1}, x_t, y_t, optional Q/R features
```

and outputs:

```text
mu_t, logvar_t, a_t, b_t, logtau_t^2
```

This answers whether the posterior parameterization and ELBO are correct.

### 8.2 Add GRU/LSTM for historical comparison

The original experiment used an LSTM that emitted parameters of a two-dimensional Gaussian edge posterior. Recreate that comparison, but with the improved marginal-preserving factorization.

### 8.3 Add Mamba/Mamba-2 as a context model

Mamba-style sequence models are plausible here because filtering is causal and may require long-range memory. However, they should be introduced as **context-assisted amortizers**, not as replacements for the explicit belief.

Recommended Mamba experiment:

```text
context_t = Mamba(context_{t-1}, embed(q^F_{t-1}, x_t, y_t))
posterior_params_t = Head(q^F_{t-1}, x_t, y_t, context_t)
q^E_t = q^F_t q^B_t
```

Required ablations:

- Head with no Mamba context.
- Mamba context reset every `k` steps.
- Predictive head using `q^F_t` only.
- Predictive head using `q^F_t + context_t`, reported separately.

Interpretation rule:

- If `q^F_t`-only prediction is good, the learned belief is carrying the information.
- If only `q^F_t + context_t` works, the model is useful but should be described as a context-augmented sequence model, not a pure learned Bayesian filter.

## 9. Posterior predictive distributions

### 9.1 One-step predictive before assimilation

This is the main predictive object for online scoring:

```text
p(y_t | D_{1:t-1}, x_t)
```

Approximate with:

```text
r_omega(y_t | q^F_{t-1}, x_t)
```

This head must not see `y_t` before predicting `y_t`.

### 9.2 Current posterior predictive after assimilation

This is useful for posterior checks:

```text
p(tilde y_t | D_{1:t}, x_t)
```

Approximate with:

```text
r_omega_current(tilde y_t | q^F_t, x_t)
```

This object can condition on `q^F_t`, which already contains information from `y_t`.

### 9.3 Model-consistent Monte Carlo predictive

For any model where sampling is possible:

```text
z_{t-1}^{(k)} ~ q^F_{t-1}
z_t^{(k)}     ~ p(z_t | z_{t-1}^{(k)})
y_t^{(k)}     ~ p(y_t | z_t^{(k)}, x_t)
```

Use these samples to:

- estimate predictive moments;
- train a predictive head by distillation;
- compare direct predictive training against model-consistent predictive training.

## 10. Evaluation plots

Every experiment should produce a standard report with:

1. observations `x_t`, `y_t`;
2. true latent `z_t` where available;
3. learned filtering mean and intervals;
4. Kalman filtering mean and intervals for linear-Gaussian experiments;
5. learned vs exact posterior variance;
6. exact vs learned edge covariance;
7. Kalman gain and learned update strength;
8. predictive distribution intervals for `y_t`;
9. calibration curves or coverage table;
10. training curves for ELBO, KL, and predictive NLL.

## 11. Suggested experiment configs

Place these in `experiments/`.

```text
experiments/linear_gaussian/00_oracle_check.yaml
experiments/linear_gaussian/01_supervised_edge_mlp.yaml
experiments/linear_gaussian/02_elbo_edge_mlp.yaml
experiments/linear_gaussian/03_elbo_edge_gru.yaml
experiments/linear_gaussian/04_elbo_edge_lstm.yaml
experiments/linear_gaussian/05_elbo_edge_mamba.yaml
experiments/linear_gaussian/06_predictive_head.yaml
experiments/linear_gaussian/07_random_qr_generalization.yaml
experiments/linear_gaussian/08_weak_observability.yaml
experiments/nonlinear/01_sine_observation.yaml
experiments/nonlinear/02_student_t_observation.yaml
```

Each config should record:

```text
seed
model_family
posterior_family
objective
data_regime
Q/R settings
sequence length
batch size
optimizer
learning rate
gradient clipping
num MC samples
logging interval
evaluation interval
checkpoint interval
```

## 12. Unit tests

Minimum tests:

| Test file | Purpose |
|---|---|
| `tests/test_kalman_scalar.py` | Standard Kalman recursion matches known values |
| `tests/test_edge_oracle.py` | Edge marginal over `z_t` equals Kalman filtering posterior |
| `tests/test_gaussian_edge_posterior.py` | `q^F q^B` gives correct joint mean/covariance/log_prob |
| `tests/test_elbo_shapes.py` | ELBO returns scalar and supports batching/time dimensions |
| `tests/test_positive_scales.py` | Learned posterior scales are always positive |
| `tests/test_predictive_no_leakage.py` | Pre-assimilation predictive head does not receive `y_t` |
| `tests/test_reproducibility.py` | Fixed seed reproduces data and initial metrics |

## 13. Initial acceptance criteria

Suggested first-pass targets for the scalar linear-Gaussian benchmark:

| Criterion | Target |
|---|---|
| Oracle edge marginal consistency | numerical error `< 1e-8` in float64 |
| Supervised MLP `filter_kl` | near zero on validation |
| Supervised MLP `edge_kl` | near zero on validation |
| ELBO MLP `filter_kl` | clearly below naive baseline |
| ELBO MLP interval coverage | close to nominal 90/95% on same-distribution data |
| Weak-observability variance | learned variance increases when `x_t ≈ 0` |
| Predictive NLL | close to exact Kalman predictive NLL in linear-Gaussian setting |
| Longer sequence evaluation | no systematic variance collapse or explosion |

Do not overfit to these exact numbers initially. Use them as sanity checks and tighten thresholds once stable baselines exist.

## 14. Main risks and mitigations

### Risk: the context model hides the posterior state

Mitigation:

- Report strict-filter and context-assisted results separately.
- Evaluate predictive heads using `q^F_t` only.
- Add context ablations.

### Risk: ELBO training collapses variance

Mitigation:

- Use variance floors.
- Monitor coverage.
- Compare to supervised distillation.
- Try KL warmup or entropy regularization only after confirming the objective implementation.

### Risk: learned posterior matches `z_t` marginal but not edge covariance

Mitigation:

- Always report `edge_kl` and edge covariance error.
- Include weak-observability tests where cross-time uncertainty matters.

### Risk: posterior predictive head cheats

Mitigation:

- Enforce an API distinction between pre-assimilation and post-assimilation inputs.
- Add tests proving `y_t` is not passed to the pre-assimilation head.

### Risk: Monte Carlo ELBO has high variance

Mitigation:

- Start with Gaussian cases where exact expectations can be used for debugging.
- Use reparameterized sampling.
- Increase MC samples only after validating shapes and signs.
- Consider antithetic samples or control variates later.

## 15. Recommended first implementation sequence

1. Create the repo structure.
2. Archive the original scripts.
3. Implement `data.py` for scalar linear-Gaussian episodes.
4. Implement `kalman.py` with standard and edge posterior outputs.
5. Implement `GaussianBelief`, `ConditionalGaussianBackward`, and `GaussianEdgePosterior`.
6. Implement closed-form Gaussian KL metrics.
7. Implement supervised edge distillation with a strict MLP cell.
8. Implement Monte Carlo edge ELBO.
9. Add the one-step predictive head.
10. Add GRU/LSTM comparison.
11. Add Mamba/Mamba-2 context model.
12. Add nonlinear/non-Gaussian experiments.

## 16. What to avoid initially

- Do not start with Mamba before the strict MLP filter works.
- Do not use an arbitrary joint flow over `(z_t, z_{t-1})` unless the filtering marginal is also explicit or approximated by a tested projection/compression step.
- Do not evaluate only RMSE; uncertainty calibration and edge covariance are central.
- Do not allow the predictive head to condition on the measurement it is supposed to predict.
- Do not conflate supervised oracle distillation with unsupervised variational recovery; both are useful but answer different questions.
