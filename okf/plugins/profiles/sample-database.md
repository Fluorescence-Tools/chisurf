---
type: Plugin Profile
title: Sample Database legacy marker
description: OKF profile for the retired root-level sample database plugin.
resource: chisurf/plugins/sample_database/
tags: [plugins, mfdb, legacy, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `sample_database` |
| Display name | `Legacy:Sample Database` |
| Categories | `Tools`, `Fluorescence` |
| Version | `1.1.0` |
| State namespace | `sample_database` |
| Local README | Missing |

This is a retired prerelease surface. All active database administration belongs in
`core/mfdb_admin` and canonical `mfdb.*` / `mfdb.v1.*` services.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| GUI | Wrapper files remain in the directory, but the manifest no longer exposes a GUI entrypoint. |
| Backend services | Wrapper files remain in the directory, but the manifest no longer exposes a service entrypoint. |
| API placeholder | `api/` exists. |
| Tests | No plugin-local tests were found in the current tree. |

The manifest is hidden/deprecated, has no entrypoints, and declares no RPC methods.
Remaining files should be treated as deletion candidates, not as a support surface.

# Data And Provenance Impact

This plugin used to overlap MFDB Admin. Because ChiSurf is prerelease, legacy support
is not required; work should move directly to `core/mfdb_admin` and old aliases should
be removed instead of documented as supported compatibility.

# Verification Surface

No plugin-local tests are needed for this retired surface. Verification should prove
the canonical `core/mfdb_admin` surface works and that no live code calls
`sample_database.*`.

# Documentation Work

- Remove remaining wrapper files when convenient.
- Keep `sample_database` hidden/deprecated until deletion.
- Do not add new `sample_database.*` aliases.
- Move any remaining useful implementation into `core/mfdb_admin`.
