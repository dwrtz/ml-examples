---
name: lambda-vm-shutdown
description: Safely finish and tear down the ml-examples Lambda SkyPilot GPU VM. Use when the user asks to shut down, terminate, destroy, stop billing, clean up, release, or end the Lambda GPU VM after syncing experiment artifacts.
---

# Lambda VM Shutdown

Use this skill to preserve outputs and terminate the disposable Lambda VM.

## Constants

- Repo root: `/Users/djwurtz/proj/ml-examples`
- Cluster name: `ml-exp`
- Remote outputs: `~/sky_workdir/outputs/`
- Local artifact root: `outputs/cloud_downloads/`

## Workflow

1. Check cluster and queued jobs:

```bash
sky status --refresh
sky queue ml-exp
```

2. If jobs are running and the user did not explicitly ask to interrupt them, report the active job ids and do not terminate yet.

3. Pull all remote outputs before teardown:

```bash
mkdir -p outputs/cloud_downloads
rsync -Pavz ml-exp:~/sky_workdir/outputs/ outputs/cloud_downloads/
```

4. Spot-check local artifacts:

```bash
find outputs/cloud_downloads -maxdepth 3 -type f | sort | tail -50
```

5. Terminate the cluster:

```bash
sky down -y ml-exp
```

6. Confirm no active cluster remains:

```bash
sky status --refresh
```

## Optional Code Checkpoint

If the user asks to check in code, review `git status --short` after artifact sync. Stage only the files intentionally changed for the workflow or experiment. Do not stage ignored `outputs/` artifacts unless the user explicitly asks.

## Safety

- Prefer `sky down`, not `sky stop`, for this workflow. The VM is disposable and artifacts should be local before teardown.
- Never assume outputs are safe until `rsync` has completed successfully.
- If `rsync ml-exp:...` fails, run `sky status --refresh` and retry before teardown.
- If SkyPilot and Lambda disagree after teardown, tell the user to verify the Lambda console and terminate any leaked instance there.
