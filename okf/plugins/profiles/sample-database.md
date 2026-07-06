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

This is a deleted prerelease surface. All active database administration belongs
in `core/mfdb_admin` and canonical `mfdb.admin`, `mfdb.*`, and `mfdb.v1.*` services.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| GUI | Deleted with the root-level plugin. |
| Backend services | Deleted with the root-level plugin; active handlers live under `mfdb.admin.backend`. |
| API placeholder | Deleted. |
| Tests | Legacy compatibility test deleted; active coverage moved to canonical MFDB Admin/FDB tests. |

The plugin directory and manifest are deleted. This profile remains only as a
retirement/tombstone note.

# Data And Provenance Impact

This plugin used to overlap MFDB Admin. Because ChiSurf is prerelease, legacy support
is not required; work should move directly to `core/mfdb_admin` and old aliases should
be removed instead of documented as supported compatibility.

# Verification Surface

No plugin-local tests are needed for this retired surface. Verification should prove
the canonical `core/mfdb_admin` surface works and that no live code calls
`sample_database.*`.

# Documentation Work

- Keep `sample_database` out of active plugin manifests and loader paths.
- Do not add new `sample_database.*` aliases.
- Move any remaining useful implementation into `core/mfdb_admin`.
