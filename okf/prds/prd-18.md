---
type: PRD
prd: "18"
title: "PRD-18: Dependency Injection + Hermetic Test Harness"
description: Makes the test suite hermetic and contracts the MFDB client while explicit database/session injection is still being completed.
status: in-progress
phase: "1"
resource: overhaul/PRD-18-dependency-injection-test-harness.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-18 makes the test suite hermetic and defines an explicit MFDB client contract, while progressively replacing module-global database/session resolution with explicit boundary injection. The autouse conftest fixture redirects the settings directory, resolved database, and object store to a per-session temporary location so no test can read or write user data. `MFDBClient.call`/`close` is exercised by integration tests against the real in-process client, so interface drift fails loudly instead of being masked by mocks. The broader dependency-injection goal is not complete yet: many RPC/API handlers still open `MFDatabase(resolve_database_path())` directly, overlapping with the open [PRD-17](prd-17.md) SessionContext work.

# Status
In-progress (re-verified against code 2026-07-05). Landed: the hermetic test harness (`test/conftest.py`) and real in-process `MFDBClient.call` contract test. **Open:** handler/session injection is not complete; `chisurf/core/mfdb/api.py` and `chisurf/plugins/core/mfdb_admin/backend/*.py` still contain many direct `MFDatabase(resolve_database_path())` calls, so the "one injected session/db boundary" DoD remains unmet.

# Relationships
- Carries [PRD-17](prd-17.md)'s SessionContext as the injected unit.
- Contracts the client `.call` fix that originated in [PRD-10](prd-10.md).
- Its hermetic harness is a prerequisite for [PRD-24](prd-24.md) standalone package tests.
- Reinforced by [PRD-25](prd-25.md) (fail-loud error policy, single RPC envelope).
- Targets the [MFDB target](/specs/mfdb.md) and [Core target](/specs/core.md); moves away from [runtime globals](/architecture/runtime-globals.md).

# Source
- Primary: `overhaul/PRD-18-dependency-injection-test-harness.md`
