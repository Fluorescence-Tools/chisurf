# Releases

ChiSurf is maintained with **one major release line per year** and optional patch releases.

## Branch Model

- `development`: next-major development (features + fixes)
- `25`, `26`, `27`, ...: per-major maintenance branches
  - after first stable tag on a major branch: **bugfix-only**
- `master` (optional): pointer to latest stable major (fast-forward to the newest maintenance branch)

## Release Policy

- At most **one new major line per year** (e.g. open `26` once per year)
- Patch releases on an existing major line are allowed (bugfix-only)

## Cutting a New Major Line

1. Create branch `26` from `development` (feature-freeze point).
2. Stabilize on `26` (bugfixes/docs only).
3. Tag stable release: `v26.0.0`.
4. Publish artifacts (pip/conda/installer).
5. Merge/fast-forward `master` to `26` (if `master` is used).

## Backporting Fixes to Older Majors

1. Implement the fix on `development` first.
2. Cherry-pick the minimal fix commit(s) onto the old major branch (e.g. `25`).
3. Tag a patch release on that branch using a new version tag (e.g. `v25.1.1`).

## Build/Publish Notes

- pip/CI builds bake a fixed version string into `chisurf/info.py` via the custom build backend.
- conda builds should set `CHISURF_VERSION` from the intended tag/version.
- Windows installers mark dev builds explicitly (name and `_dev` filename suffix).
