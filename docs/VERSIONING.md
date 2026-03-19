# Versioning

ChiSurf uses a simplified versioning scheme that remains **PEP 440** compatible for pip/conda.

## Public Version Format

- Release line (major): `YY` (year)
- Major version: `XX`
- Minor version: `YY`

Stable releases are tagged and published as:

- `YY.XX.YY`

Examples:

- `26.0.0` (first stable release in 2026)
- `26.1.0` (next major feature release in 2026)
- `26.1.1` (bugfix release)

## Dev Versions

Dev builds are derived from git metadata (and are therefore unique per commit):

- `YY.devZZZ`

Where `ZZZ` is the number of commits since the last matching tag.

Notes:

- A tagged commit builds exactly the tag version (e.g., `26.0.0`).
- A non-tagged commit builds a dev version that sorts *after* the base tag (e.g., `26.dev123`).

## Tag Naming

- Tags use a leading `v`: `v26.0.0`

## Overrides

- Set `CHISURF_VERSION` to force a specific version string during builds.
