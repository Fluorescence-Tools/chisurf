# Releases

ChiSurf is maintained with **one major release line per year** and optional minor releases.

See [VERSIONING.md](VERSIONING.md) for version format details (YY.X scheme, pre-release suffixes, dev versions, tag naming).

## Branch Model

- `development`: next-major development (features + fixes)
- `25`, `26`, `27`, ...: per-major maintenance branches
  - after first stable tag on a major branch: **bugfix-only**
- `master` (optional): pointer to latest stable major (fast-forward to the newest maintenance branch)

## Release Policy

- At most **one new major line per year** (e.g. open `26` once per year)
- Minor releases on an existing major line are allowed (bugfix + improvements)
- No point releases; fixes go into the next minor (`26.1`)

## Pre-Release Stages

See [VERSIONING.md](VERSIONING.md) for pre-release format (`aN`, `bN`, `rcN`) and tag naming convention.

1. **Dev** (`26.devXXX`): every commit on development, auto-versioned from git
2. **Alpha** (`26.1a1`): internal testing, core workflows functional
3. **Beta** (`26.1b1`): feature-complete, external beta testers
4. **Release Candidate** (`26.1rc1`): no known critical bugs, final polish
5. **Stable** (`26.1`): production-ready

## Cutting a New Major Line

See [VERSIONING.md](VERSIONING.md) for tag format details.

1. Create branch `26` from `development` (feature-freeze point).
2. Stabilize on `26` (bugfixes/docs only).
3. Tag alpha: `v26.1a1`. Iterate as needed (`v26.1a2`, ...).
4. Tag beta: `v26.1b1`. Iterate as needed.
5. Tag RC: `v26.1rc1`. Iterate as needed.
6. Tag stable release: `v26.1`.
7. Publish artifacts (pip/conda/installer).
8. Merge/fast-forward `master` to `26` (if `master` is used).

## Backporting Fixes to Older Majors

1. Implement the fix on `development` first.
2. Cherry-pick the minimal fix commit(s) onto the old major branch (e.g. `25`).
3. Tag a minor release on that branch (e.g. `v25.1`).

## Automated Release Workflow

ChiSurf uses GitHub Actions (`pixi-ci.yml`) to automate the build and release process.

### Triggering a Release via Tags

1.  **Draft a Tag**: Following the naming convention in `VERSIONING.md`, created a new git tag.
    - Example: `git tag -a v26.1 -m "Stable release 26.1"`
    - Example: `git tag -a v26.1a1 -m "Alpha 1 release"`
2.  **Push the Tag**: `git push origin v26.1`
3.  **Automated Build**: GitHub Actions will trigger on the tag. It will:
    - Build conda packages for Linux, macOS, and Windows.
    - Build platform-specific installers (DMG, AppImage, EXE).
    - Create a GitHub Release.
    - Upload all artifacts to the release.
    - **Mark as Pre-release**: If the tag contains `a`, `b`, or `rc`, the GitHub Release is automatically marked as a pre-release.

### Triggering a Manual Release

You can manually trigger a release build using the `workflow_dispatch` trigger in the GitHub Actions UI:

1.  Go to **Actions** -> **Build and Release** workflow.
2.  Click **Run workflow**.
3.  Choose the branch (e.g., `main` or `development`).
4.  Select the **Release Type** (dev, alpha, beta, rc, stable). This determines the versioning metadata for the build.

## Build/Publish Notes

- **Version Persistence**: The `rattler-recipe/generate_version.py` script dynamically computes the version from git tags.
- **Installer Naming**: Installers are versioned (e.g., `ChiSurf-Windows-Setup-26.1.exe`).
- **Release Notes**: GitHub Release notes are automatically generated from commit messages between the current and previous tag.
