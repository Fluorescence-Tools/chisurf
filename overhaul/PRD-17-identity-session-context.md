# PRD-17: Canonical Identity / Session Context (Architecture A)

## Goal

Resolve "who is the active user" and "which database" **once**, at the boundary,
into a single `SessionContext` threaded explicitly through registration, browse,
and ownership code — eliminating the divergent identity resolution that caused
ownership/visibility bugs.

## Evidence (why)

"Active user" is resolved two ways: `register_*` stamps
`cs_settings["mfdb"]["default_user_id"]`; RPC handlers resolve the **auth
principal** (anonymous for the in-process GUI). The mismatch made the dataset
browser's "Mine" return 0 (the query resolved a different user than registration
stamped) and forced ad-hoc "anonymous → default user" fallbacks in
`datasets.browse` and `datasets.open`.

## Design

- A `SessionContext` dataclass: `user_id`, `db` (handle/path), `is_admin`,
  `groups`, `auth`. Constructed **once** per entry point: GUI launch (from settings
  / login) and RPC dispatch (from the auth principal, falling back to the
  configured default user *in one place*).
- All MFDB read/write APIs take the context (or its `user_id`/`db`) explicitly:
  `register_result(..., session=ctx)`, `browse_datasets(..., session=ctx)`,
  ownership stamping, study/protocol scoping.
- Remove independent `_resolve_active_user_id()` calls scattered across
  `result_registry`, `tttr_setup_utils`, handlers — they all consult the one
  resolver behind the context.
- One **canonical current-user resolver** used by both reads and writes, so an
  artifact stamped by user X is always found under X's "Mine".

## Tasks

1. Define `SessionContext` + a single `resolve_session(auth=None)` (settings ⊕ auth)
   in `chisurf/core/mfdb/session.py`.
2. Thread it through `register_result`/`register_raw_measurement`, browse/open
   handlers, ownership, and setup/sample resolution; delete the duplicate
   resolvers.
3. RPC dispatch builds the context once from auth (with the default-user fallback
   centralized here, not per handler).
4. Tests: write-then-read-as-same-user round trips under both logged-in and
   no-login modes; the two-user scoping still holds; no handler re-resolves
   identity independently.

## Definition of Done

- [ ] One `SessionContext` + one current-user resolver; no module re-resolves
      identity on its own.
- [ ] Registration owner and browse "own" scope always agree (no anonymous-fallback
      patches remain).
- [ ] Tests cover logged-in and no-login paths; two-user scoping intact.

## Definition of Clean

DI (context passed, not module-global); behavior-asserting tests (write→read same
user); no swallowed identity mismatches.

## Relationship

Subsumes the per-handler anonymous-fallback patches added in PRD-10. Pairs with
PRD-18 (DI) — the context is the thing injected.
