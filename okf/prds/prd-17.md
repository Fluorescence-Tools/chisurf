---
type: PRD
prd: "17"
title: "PRD-17: Canonical Identity / Session Context"
description: Resolves the active user and target database once at the boundary into a single SessionContext threaded explicitly through registration, browse, and ownership code.
status: in-progress
phase: "1"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-17 eliminates divergent identity resolution — where registration stamped one user while read handlers resolved another — that caused ownership and visibility bugs (for example, "Mine" returning zero results). It introduces a single `SessionContext` dataclass carrying `user_id`, database handle, admin flag, groups, and auth principal, constructed once per entry point (GUI launch or RPC dispatch). All MFDB read and write APIs take the context explicitly, and the scattered per-module current-user resolvers are removed in favor of one canonical resolver used by both reads and writes.

# Status
In-progress (re-verified against code 2026-07-05). The canonical current-user resolver landed and the "Mine returns 0" ownership/visibility bug is fixed (registration owner and browse scope agree). **Open:** the "one `SessionContext`, constructed once per entry point and threaded explicitly" design is not realized — `resolve_session()` has no call sites and ~10 modules still resolve identity independently, so the DoD ("no module re-resolves identity on its own") is unmet. Corroborated by assessment [SV-03](/specs/assessment.md#sv-03).

# Goal
Resolve "who is the active user" and "which database" **once**, at the boundary, into a single `SessionContext` threaded explicitly through registration, browse, and ownership code — eliminating the divergent identity resolution that caused ownership/visibility bugs.

# Evidence (why)
"Active user" is resolved two ways: `register_*` stamps `cs_settings["mfdb"]["default_user_id"]`; RPC handlers resolve the **auth principal** (anonymous for the in-process GUI). The mismatch made the dataset browser's "Mine" return 0 (the query resolved a different user than registration stamped) and forced ad-hoc "anonymous → default user" fallbacks in `datasets.browse` and `datasets.open`.

# Design
- A `SessionContext` dataclass: `user_id`, `db` (handle/path), `is_admin`, `groups`, `auth`. Constructed **once** per entry point: GUI launch (from settings / login) and RPC dispatch (from the auth principal, falling back to the configured default user *in one place*).
- All MFDB read/write APIs take the context (or its `user_id`/`db`) explicitly: `register_result(..., session=ctx)`, `browse_datasets(..., session=ctx)`, ownership stamping, study/protocol scoping.
- Remove independent `_resolve_active_user_id()` calls scattered across `result_registry`, `tttr_setup_utils`, handlers — they all consult the one resolver behind the context.
- One **canonical current-user resolver** used by both reads and writes, so an artifact stamped by user X is always found under X's "Mine".

# Tasks
1. Define `SessionContext` + a single `resolve_session(auth=None)` (settings ⊕ auth) in `chisurf/core/mfdb/session.py`.
2. Thread it through `register_result`/`register_raw_measurement`, browse/open handlers, ownership, and setup/sample resolution; delete the duplicate resolvers.
3. RPC dispatch builds the context once from auth (with the default-user fallback centralized here, not per handler).
4. Tests: write-then-read-as-same-user round trips under both logged-in and no-login modes; the two-user scoping still holds; no handler re-resolves identity independently.

# Definition of Done
- [ ] One `SessionContext` + one current-user resolver; no module re-resolves identity on its own.
- [ ] Registration owner and browse "own" scope always agree (no anonymous-fallback patches remain).
- [ ] Tests cover logged-in and no-login paths; two-user scoping intact.

# Definition of Clean
DI (context passed, not module-global); behavior-asserting tests (write→read same user); no swallowed identity mismatches.

# Relationships
- Subsumes the per-handler anonymous-fallback patches added in [PRD-10](prd-10.md).
- Pairs with [PRD-18](prd-18.md) — the SessionContext is the unit that gets dependency-injected.
- The injected identity/config becomes a prerequisite for [PRD-24](prd-24.md) package extraction.
- Targets the [MFDB target](/specs/mfdb.md); reduces reliance on [runtime globals](/architecture/runtime-globals.md).
