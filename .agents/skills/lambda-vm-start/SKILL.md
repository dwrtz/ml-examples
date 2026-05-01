---
name: lambda-vm-start
description: Start, verify, or refresh the ml-examples Lambda Cloud GPU VM managed by SkyPilot. Use when the user asks to spin up, launch, start, set up, provision, resume, sync, or check the Lambda/SkyPilot GPU VM for experiments in this repository.
---

# Lambda VM Start

Use this skill to make the disposable Lambda GPU VM ready for experiment work.

## Constants

- Repo root: `/Users/djwurtz/proj/ml-examples`
- Cluster name: `ml-exp`
- SkyPilot YAML: `cloud/lambda_rtx6000_dev.sky.yaml`
- Current accelerator in the YAML: `A10:1`
- Remote workdir: `~/sky_workdir`
- Output download root: `outputs/cloud_downloads/`

The YAML name is historical. Read the YAML before changing accelerators; this account currently exposes `A10:1` as the cheapest launchable 1x GPU through SkyPilot.

## Workflow

1. Work from the repository root.
2. Check local prerequisites:

```bash
test -f ~/.lambda_cloud/lambda_keys
chmod 600 ~/.lambda_cloud/lambda_keys
command -v sky || uv tool install --with pip 'skypilot[lambda]'
sky check
```

3. Check current cluster state:

```bash
sky status --refresh
```

4. If `ml-exp` is already `UP`, sync the latest local workspace and do not relaunch:

```bash
sky exec ml-exp --workdir . true
```

5. If `ml-exp` is missing, stopped, failed, or needs dependency/setup changes, launch from the YAML:

```bash
sky launch -y -c ml-exp cloud/lambda_rtx6000_dev.sky.yaml
```

6. Verify setup succeeded:

```bash
sky logs ml-exp
sky status --refresh
```

Look for JAX reporting a CUDA device and the repo test suite passing.

## Accelerator Checks

Use these checks before changing the YAML:

```bash
sky gpus list --infra lambda --all
sky gpus list A10 --infra lambda --all-regions
sky gpus list A6000 --infra lambda --all-regions
sky gpus list RTX6000 --infra lambda --all-regions
```

Only edit `resources.accelerators` when the requested GPU is exposed by SkyPilot for Lambda. If switching accelerators, also update experiment commands that use `--gpus=<GPU>:<COUNT>`.

## Safety

- Treat local code as canonical. Do not edit source files on the VM unless explicitly recovering remote changes.
- Preserve `autostop.down: true` unless the user asks otherwise.
- Do not run `sky down` from this skill. Use `lambda-vm-shutdown` for teardown.
