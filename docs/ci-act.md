# Local CI With `act`

This repository now mirrors the GitHub Actions Linux test job locally via [`act`](https://github.com/nektos/act). Use it to validate Pixi environments before pushing to `development`.

## Prerequisites

- Docker Desktop running with Linux containers (WSL 2 backend on Windows).
- `act` v0.2.55 or newer installed on your PATH.
- Adequate disk space (~15 GB) for the `catthehacker/ubuntu:full-latest` base image.

## Quick Smoke Test

Run the lightweight workflow dedicated to local verification:

```bash
act workflow_dispatch \
  -W .github/workflows/pixi-act-test.yml \
  -j act-linux \
  -P ubuntu-latest=catthehacker/ubuntu:full-latest
```

The job installs Pixi, builds editable modules, and executes `pixi run test` under `xvfb`. Use `--container-architecture linux/amd64` if Docker defaults to another platform.

## Running the Full CI Linux Job

To reproduce the Linux matrix entry from `Pixi Tests & Quality` locally:

```bash
act push \
  -W .github/workflows/test-coverage.yml \
  -j tests \
  -P ubuntu-latest=catthehacker/ubuntu:full-latest
```

Only the Linux variant runs under `act`; macOS and Windows jobs remain remote-only. Pass `--artifact-server-path <path>` to persist coverage artifacts if required.

## Troubleshooting

- If Pixi installation fails due to missing build tools, re-run after `docker image prune -f` to reclaim space.
- Cached Pixi environments live under `.pixi`; remove the directory to force a clean install.
- When Docker Desktop restarts, re-run `act` so that `xvfb` gets a fresh shared-memory mount.
