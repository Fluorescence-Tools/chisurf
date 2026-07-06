---
type: PRD
prd: "18"
title: "PRD-18: Dependency Injection + Hermetic Test Harness"
description: Makes the test suite hermetic and contracts the MFDB client while explicit database/session injection is still being completed.
status: in-progress
phase: "1"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-18 makes the test suite hermetic and defines an explicit MFDB client contract, while progressively replacing module-global database/session resolution with explicit boundary injection. The autouse conftest fixture redirects the settings directory, resolved database, and object store to a per-session temporary location so no test can read or write user data. `MFDBClient.call`/`close` is exercised by integration tests against the real in-process client, so interface drift fails loudly instead of being masked by mocks. The broader dependency-injection goal is not complete yet: many RPC/API handlers still open `MFDatabase(resolve_database_path())` directly, overlapping with the open [PRD-17](prd-17.md) SessionContext work.

# Status
In-progress (re-verified against code 2026-07-05). Landed: the hermetic test harness (`test/conftest.py`) and real in-process `MFDBClient.call` contract test. **Open:** handler/session injection is not complete; `chisurf/core/mfdb/api.py` and `chisurf/plugins/core/mfdb_admin/backend/*.py` still contain many direct `MFDatabase(resolve_database_path())` calls, so the "one injected session/db boundary" DoD remains unmet.

# Goal
Stop resolving the database/dictionary/user via module-global functions re-imported into many namespaces; pass them explicitly. Make the test suite **hermetic** so no test can touch the real user database, and define an explicit client interface tested against the *real* in-process client.

# Evidence (why)
- `resolve_database_path` is `from … import`-bound into `services`, `setup_services`, `api`, `database_resolver`. Patching one missed `api`, so the setup-handler tests read the **real** `~/.chisurf/...db` and polluted each other (a leaked `rpc_test_setup` row made them "pass"). Repros during development mutated the live DB.
- `MFDBClient` exposed `_call` but callers used `.call` → `AttributeError` swallowed → the dataset browser silently showed 0. Tests passed because they injected a **mock** client that had `.call`.

# Design
1. **Inject the DB/session** ([PRD-17](prd-17.md) `SessionContext`) into handlers and repository entry points instead of each calling `resolve_database_path()`. One resolution site at the boundary.
2. **Hermetic test harness:** an autouse `conftest` fixture (project-wide for `test/` and plugin tests) that points the settings dir + resolved DB + object store at a per-session temp location. No test can read or write the user DB. (Removes the whole class of cross-file pollution and protects user data.)
3. **Explicit client Protocol:** `MFDBClientBase`/`MFDBClient` declare `call`, `close` as the public contract; integration tests drive the **real** in-process client (not a mock) so interface drift (`.call` vs `_call`) fails loudly.
4. Where module-global resolution must stay (legacy), funnel it through one indirection that the harness overrides.

# Tasks
1. Project-wide autouse fixture redirecting settings/db/object-store to temp.
2. Add public `call` to `MFDBClient`; an integration test exercising the real client end-to-end (browse + open) — would have caught the `.call` bug.
3. Refactor handlers to take the session/db from [PRD-17](prd-17.md) rather than re-resolving.
4. Audit + remove `from … import resolve_database_path` namespace-binding that makes patching fragile; prefer `database_resolver.resolve_database_path()` calls or injected paths.
5. Tests: full-suite run is order-independent (no cross-file pollution); a guard test asserts the real user DB path is never opened during tests.

# Definition of Done
- [ ] Autouse harness: no test touches the real user DB/object store; suite is order-independent.
- [ ] `MFDBClient.call` is the public contract; an integration test uses the real client.
- [ ] Handlers take an injected session/db; fragile namespace-bound resolution removed.

# Definition of Clean
DI over monkeypatching; integration over mock-only; behavior-asserting tests; no silent swallow of interface errors.

# Relationships
- Carries [PRD-17](prd-17.md)'s SessionContext as the injected unit.
- Contracts the client `.call` fix that originated in [PRD-10](prd-10.md).
- Its hermetic harness is a prerequisite for [PRD-24](prd-24.md) standalone package tests.
- Reinforced by [PRD-25](prd-25.md) (fail-loud error policy, single RPC envelope).
- Targets the [MFDB target](/specs/mfdb.md) and [Core target](/specs/core.md); moves away from [runtime globals](/architecture/runtime-globals.md).
