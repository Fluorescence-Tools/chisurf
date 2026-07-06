---
type: PRD
prd: "24"
title: "PRD-24: Extract MFDB into a Standalone Package"
description: Moves MFDB (schema, dictionary, generator, repository/API, server, admin backend) out of chisurf into an independent module with a stable public API and no chisurf imports.
status: planned
phase: "5"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-24 extracts MFDB — schema and dictionary and generator, repository and API, object store, result registry, auth, server, and admin backend — out of `chisurf` into an independent `modules/mfdb` package with a stable public API and no `chisurf` imports, forcing a clean boundary and enabling reuse and independent testing. Settings and paths the package needs (database path, object-store root, default user) are injected rather than read from chisurf settings, and a thin chisurf adapter wires the application's config and GUI to the package API. The package is independently installable and carries its own hermetic test suite. It is sequenced last, after the interfaces it depends on have stabilized, so the move preserves stable interfaces instead of relocating churn.

# Status
Planned (phase 5, STATUS TABLE authoritative). Deferred to the end of the architecture track.

# Goal
Move MFDB (schema, dictionary, generator, repository/API, server, admin backend) out of `chisurf` into an independent vendored `modules/mfdb` package with a stable public API and **no `chisurf` imports**. This forces a clean boundary, enables reuse and independent testing, and prepares a later move to its own dedicated repository.

# Background
This was originally listed as deferred cleanup: move MFDB out of `chisurf` into `modules/` — own the schema, repository/API layer, MFDB server, and admin backend — after the current overhaul is basically finished so the extraction can preserve stabilized interfaces instead of moving churn. The current prerelease direction is a harder cut: vendor the package under `modules/mfdb` first, then remove the remaining ChiSurf imports and publish it as its own repository.

# When
**Last** — after the operation/transformer spine ([PRD-11](prd-11.md) / [PRD-16](prd-16.md)), the identity/DI work ([PRD-17](prd-17.md) / [PRD-18](prd-18.md)), and the dictionary/migration cleanup ([PRD-19](prd-19.md)) have stabilized the interfaces. Extracting earlier just moves churn.

# Design
- `modules/mfdb/` owns: `schema` + `.dic` + generator, `repository`/`api`, `object_store`, `result_registry`, `auth`, `server`, and the admin backend. The GUI admin *frontend* may stay in chisurf (Qt) but talks only through the package's RPC/API.
- **No `chisurf` imports** in the package. Settings/paths it needs (db path, object-store root, default user) are injected ([PRD-17](prd-17.md) `SessionContext` / explicit config), not read from `chisurf.core.settings`.
- A thin `chisurf` adapter wires chisurf settings/GUI to the package API.
- The package is independently installable and testable (its own hermetic test suite from [PRD-18](prd-18.md)).

# Tasks
1. Inventory and cut the chisurf→mfdb dependency edges; replace `chisurf.core.settings` reads with injected config (depends on [PRD-17](prd-17.md) / [PRD-18](prd-18.md)).
2. Move the modules; keep import shims in `chisurf.core.mfdb` only as a transitional re-export, then delete.
3. Package metadata + standalone test run (uses the hermetic harness).
4. chisurf adapter: config/GUI ↔ package API.
5. CI: the package builds and tests on its own.

# Definition of Done
- [x] MFDB source package moved to `modules/mfdb/src/mfdb`; `chisurf.core.mfdb` remains only as a transitional facade.
- [ ] `modules/mfdb` is a standalone package with no `chisurf` imports; injected config.
- [ ] chisurf uses it via a thin adapter; transitional shims removed.
- [ ] Package builds and its hermetic tests pass independently.

# Definition of Clean
No `chisurf` imports in the package; config injected not global; the hermetic test harness ([PRD-18](prd-18.md)) travels with the package; delete shims, don't alias.

# Relationships
- Capstone of the architecture track; requires injected identity/config from [PRD-17](prd-17.md) and the injected database plus hermetic tests from [PRD-18](prd-18.md).
- Benefits from the self-contained dictionary-driven schema of [PRD-19](prd-19.md).
- Targets the [MFDB target](/specs/mfdb.md); the package talks over the [RPC target](/specs/rpc.md) and removes [runtime globals](/architecture/runtime-globals.md) dependencies.
