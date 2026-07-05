---
type: PRD
prd: "24"
title: "PRD-24: Extract MFDB into a Standalone Package"
description: Moves MFDB (schema, dictionary, generator, repository/API, server, admin backend) out of chisurf into an independent module with a stable public API and no chisurf imports.
status: planned
phase: "5"
resource: overhaul/PRD-24-mfdb-package-extraction.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-24 extracts MFDB — schema and dictionary and generator, repository and API, object store, result registry, auth, server, and admin backend — out of `chisurf` into an independent `modules/mfdb` package with a stable public API and no `chisurf` imports, forcing a clean boundary and enabling reuse and independent testing. Settings and paths the package needs (database path, object-store root, default user) are injected rather than read from chisurf settings, and a thin chisurf adapter wires the application's config and GUI to the package API. The package is independently installable and carries its own hermetic test suite. It is sequenced last, after the interfaces it depends on have stabilized, so the move preserves stable interfaces instead of relocating churn.

# Status
Planned (phase 5, STATUS TABLE authoritative). Deferred to the end of the architecture track.

# Relationships
- Capstone of the architecture track; requires injected identity/config from [PRD-17](prd-17.md) and the injected database plus hermetic tests from [PRD-18](prd-18.md).
- Benefits from the self-contained dictionary-driven schema of [PRD-19](prd-19.md).
- Targets the [MFDB target](/specs/mfdb.md); the package talks over the [RPC target](/specs/rpc.md) and removes [runtime globals](/architecture/runtime-globals.md) dependencies.

# Source
- Primary: `overhaul/PRD-24-mfdb-package-extraction.md`
