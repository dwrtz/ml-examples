---
name: lambda-run-experiment
description: Run GPU-accelerated ml-examples experiments on the Lambda SkyPilot VM and sync artifacts back. Use when the user asks to run, rerun, launch, monitor, cancel, or retrieve outputs for experiments on the Lambda GPU VM.
---

# Lambda Run Experiment

Use this skill for the loop: sync local workspace to the VM, run a GPU job with explicit GPU scheduling, monitor logs, and pull artifacts back.

## Constants

- Repo root: `/Users/djwurtz/proj/ml-examples`
- Cluster name: `ml-exp`
- Current GPU demand flag: `--gpus=A10:1`
- Remote workdir: `~/sky_workdir`
- Remote outputs: `~/sky_workdir/outputs/`
- Local artifact root: `outputs/cloud_downloads/`

Ad hoc `sky exec` commands must include `--gpus=A10:1` for GPU experiments. Without this flag, this SkyPilot version may run the job without CUDA access even on a GPU VM.

## Preflight

1. Confirm the cluster is up:

```bash
sky status --refresh
```

2. If `ml-exp` is not `UP`, use the `lambda-vm-start` workflow before running experiments.
3. Inspect the requested experiment command. Prefer existing Makefile targets and checked-in experiment YAMLs.
4. Choose a unique output directory under `outputs/`, usually with a run name that includes the experiment purpose, step count, and date or timestamp.

## Run Commands

For a foreground run:

```bash
sky exec ml-exp --workdir . --gpus=A10:1 '<experiment command>'
```

For a detached run:

```bash
sky exec -d ml-exp --workdir . --gpus=A10:1 '<experiment command>'
```

Example nonlinear learned sweep:

```bash
sky exec ml-exp --workdir . --gpus=A10:1 \
  'make sweep-nonlinear-learned \
     NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
     NONLINEAR_LEARNED_MODELS=structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,direct_mixture_k2_joint_iwae_h4_k32 \
     NONLINEAR_LEARNED_SEEDS=321,322,323 \
     NONLINEAR_LEARNED_STEPS=1000 \
     NONLINEAR_LEARNED_DIR=outputs/lambda_probe_1000'
```

For a quick CUDA smoke check:

```bash
sky exec ml-exp --workdir . --gpus=A10:1 \
  'JAX_PLATFORMS=cuda uv run python -c "import jax; print(jax.devices())"'
```

## Monitor And Control

Use these commands for detached or long runs:

```bash
sky queue ml-exp
sky logs ml-exp
sky logs ml-exp <JOB_ID>
sky cancel ml-exp <JOB_ID>
```

If a run fails before Python starts, check whether shell quoting split the remote command. In this SkyPilot version, the entrypoint should be a single quoted command after SkyPilot flags.

## Pull Artifacts

Pull artifacts after every important run:

```bash
mkdir -p outputs/cloud_downloads
rsync -Pavz ml-exp:~/sky_workdir/outputs/<RUN_NAME>/ outputs/cloud_downloads/<RUN_NAME>/
```

To pull all outputs:

```bash
mkdir -p outputs/cloud_downloads
rsync -Pavz ml-exp:~/sky_workdir/outputs/ outputs/cloud_downloads/
```

After pulling, inspect key files such as `metrics.csv`, generated reports, plots, summaries, and logs before summarizing results.

## Iteration Rules

- Edit source code locally, not on the VM.
- Use `sky exec --workdir . --gpus=A10:1 ...` to sync before each run.
- Use `sky launch -y -c ml-exp cloud/lambda_rtx6000_dev.sky.yaml` after dependency, lockfile, Python version, or YAML setup changes.
- Do not shut down the VM from this skill unless the user explicitly asks to finish and teardown; use `lambda-vm-shutdown` for that workflow.
