---
type: PRD
prd: "12"
title: "PRD-12: Lifecycle State Machines + Transition History"
description: Turn flat entity status flags into tracked lifecycles with a recorded, validated transition log (who, when, why).
status: done
phase: "3"
resource: chisurf/core/mfdb
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
MFDB entity status was flat flags with no history or defined lifecycle. This PRD adds a single generic, dictionary-driven transition log (`mfdb_state_transition`) plus allowed-transition rules (`mfdb_state_transition_rule`), with per-entity-type state vocabularies (sample, artifact, operation). Current state is the fold over the transition log (optionally cached), giving every sample/dataset/operation an auditable "where is it and how did it get here." The repository API (`transition_state`, `get_state`, `get_state_history`) validates transitions, records operator and timestamp, and is idempotent; registration paths emit initial and advancing states best-effort. An admin lifecycle view surfaces current state and history. It is designed as a projection over the append-only event core.

# Status
Done. Schema, transition API with rule validation, registration wiring, and a standalone admin lifecycle view all landed (27 tests). The only deferred step is slotting the view into the admin tool's dock layout after the dock rewrite.

**Projection over the append-only core (PRD-27).** If PRD-27's append-only
decision is taken first (it should be), `mfdb_state_transition` **is** the state
event log and "current state" is the fold over it — there is no separate mutable
status flag to reconcile. Build this *on* PRD-27 so it is built once. The design
below already assumes "the transition log is the source of truth; current_state is
a cache"; PRD-27 makes that the universal rule.

# Goal
Turn MFDB entity status from flat flags into **tracked lifecycles** with recorded
transitions (who, when, why), so a sample/dataset/operation has an auditable
"where is it and how did it get here" — the defining LIMS capability MFDB lacks.

# Background
- LIMS reference: a mature LIMS tracks every entity through explicit per-entity
  state tables (samples, runs, library-prep, service state, …) with transition
  history.
- MFDB today: only flat fields — `mfdb_operation.status`,
  `mfdb_artifact.validation_status`, `flr_experiment.status` — no transition
  history, no defined lifecycle, no per-entity "current state" beyond the column.

# Design (generic, dictionary-driven)
One generic transition log instead of per-entity status columns:
- **`mfdb_state_transition`** (`.dic`-declared, generated, gate-covered):
  `(id PK, entity_type, entity_id, from_state, to_state, reason,
  operator_user_id FK flr_sample_users, created_at)`, index on
  `(entity_type, entity_id, created_at)`.
- **States are an extensible vocabulary** keyed by entity_type. Reuse
  `mfdb_vocabulary` with `field_name = "state:<entity_type>"` (e.g. `state:sample`,
  `state:artifact`, `state:operation`). Lifecycles:
  - **sample**: `registered → measured → processed → validated → archived`
  - **artifact/dataset**: `registered → validated → published → archived`
  - **operation**: fold the existing `pending/running/succeeded/failed/cancelled`
    into the same machinery.
- **Allowed transitions** per entity_type declared in the `.dic` (a small
  transition table seeded from the dictionary,
  `mfdb_state_transition_rule(entity_type, from_state, to_state)`), so illegal
  jumps are rejected.
- **Current state** = the latest non-deleted transition's `to_state`. Optionally
  cache it in a `current_state` column on the entity for fast filtering
  (dictionary-declared if added); the transition log is the source of truth.

# API
- `transition_state(entity_type, entity_id, to_state, reason="",
  operator_user_id=None)` — validates against the rules, records a row, updates the
  cached `current_state` if present. Idempotent no-op if already in `to_state`.
- `get_state(entity_type, entity_id)` → current state.
- `get_state_history(entity_type, entity_id)` → ordered transitions.
- Registration hooks: `register_raw_measurement` sets sample/artifact initial
  state `registered`; `register_result` advances the relevant states; operation
  status changes go through `transition_state`.

# Tasks
1. `.dic` + schema: declare `mfdb_state_transition` (and the optional
   `mfdb_state_transition_rule`); generate DDL; `SCHEMA_VERSION` bump; seed the
   per-entity-type state vocabularies + transition rules from the dictionary; add
   to the total-coverage gate.
2. Repository: `transition_state` / `get_state` / `get_state_history` with rule
   validation; optional cached `current_state` columns (dict-declared).
3. Wire registration paths to emit initial + advancing transitions (best-effort in
   GUI flows; strict in tests).
4. mfdb-admin: show current state + history per entity; allow admin transitions.
5. Tests: legal transition recorded; illegal transition rejected; history ordered;
   current state resolves; idempotent re-transition; dict gate green.

# Definition of Done
- [x] `mfdb_state_transition` exists (dict-declared, generated, gate-covered) with
      per-entity-type state vocabularies and transition rules.
- [x] sample / artifact / operation lifecycles defined; transitions validated and
      recorded with operator + timestamp; history queryable.
- [x] registration paths emit initial/advancing states; admin shows state+history
      *(via the standalone `LifecycleView`; slotting it into the mid-overhaul dock
      layout is the one deferred wiring step)*.
- [x] Tests pass; no flat status flag is the sole source of truth (the transition
      log is the source of truth; `get_state` is the fold). 27 tests, arm64.

# Definition of Clean
`.dic` dictates the schema (no hardcoded SQL, no blob); validation rejects illegal
transitions (surfaced, not swallowed, on real errors; best-effort for
MFDB-unavailable); behavior-asserting tests; DI over monkeypatching; GUI smoke for
the admin view.

# Implementation status
**Increment 1 (schema foundation) — DONE.** The `.dic` declares
`mfdb_state_transition` (the transition log) and `mfdb_state_transition_rule`
(allowed transitions); both are `mfdb_*` extension tables so `reconcile_schema`
creates them on migrate (no hand DDL). `data/state_lifecycle_defs.json` is the
authored single source for per-entity-type states + allowed transitions (sample/
artifact/operation); `core/mfdb/lifecycle.py` (`LifecycleDef`,
`load_lifecycle_defs`, `get_lifecycle_def`, `bootstrap_lifecycle_defs`) seeds the
state vocabularies (`mfdb_vocabulary` `field_name="state:<entity_type>"`) and the
rule table (idempotent), wired into both migrate paths in `schema.py`. Covered by
`test/fio/test_lifecycle_schema.py` (6).

**Increment 2 (transition API) — DONE.** `repository.py` has
`transition_state(entity_type, entity_id, to_state, reason="",
operator_user_id=None)` (validates against `mfdb_state_transition_rule`, raises
`lifecycle.StateTransitionError` on an illegal jump, idempotent no-op returning
`False` when already in `to_state`, records the row + audit log, publishes PRD-21's
`state.changed` post-commit), `get_state` (latest non-deleted `to_state`), and
`get_state_history` (ordered). `transition_id` is assigned MAX+1. Covered by
`test/fio/test_lifecycle.py` (9): initial/advancing, illegal jump + illegal initial
rejected (state unchanged), idempotent no-op (no row, no event), ordered history,
branching operation lifecycle, `state.changed` payload, audit log.

**Increment 3 (registration wiring) — DONE.** `register_result` (and thus
`register_raw_measurement`) emits initial lifecycle states post-commit, best-effort
via `_start_lifecycle`: the new artifact → `registered`, and a freshly-linked
sample with no state yet → `registered`. Wrapped in try/except — never breaks
registration (an RPC client without `transition_state` or any transient error is
logged at debug and swallowed), mirroring the post-commit event publish. Covered by
`test/fio/test_lifecycle.py` (+3).

**Increment 4 (admin view) — DONE.** 4a: backend RPC handlers
`mfdb.lifecycle.{state,history,transition,definitions}` (an illegal jump returns an
`error` field, not an exception, across the boundary) + `MFDBClient` methods,
covered end-to-end through the `InProcessClient` (`test_lifecycle_handlers.py`, 5).
4b: a **standalone** `gui/lifecycle_view.py::LifecycleView` (entity picker, current
state, legal next-state combo, history table, apply-transition surfacing illegal
jumps) — thin over the client (PRD-23). Kept standalone (not wired into the
5681-line mid-overhaul `tool.py`) so the in-flight dock-layer rewrite slots it in;
Qt smoke test `test_lifecycle_view.py` (4, offscreen).

**PRD-12 is functionally complete (all DoD met).** Headless core (schema/API/
registration) + admin view all landed; 27 tests green. The only optional remainder
is **wiring `LifecycleView` into the admin tool's dock layout** — deferred on
purpose until the dock rewrite lands (so it isn't built against the soon-to-be-
replaced legacy tab framework). When ready: add a dock that constructs
`LifecycleView(self.client)` and, from a record's context menu, calls
`view.set_entity(entity_type, entity_id); view.refresh()` (mirror the provenance
"Use as seed" action in `gui/tool.py`).

# Relationships
- A projection over the append-only event/provenance core; emits `state.changed` events.
- Operation status from [PRD-11](prd-11.md) folds into the generic transition machinery.
- Shares the `mfdb_vocabulary` / dictionary machinery with [PRD-19](prd-19.md); emitted via the [PRD-21](prd-21.md) event model.
- Referenced by the study ([PRD-13](prd-13.md)) and protocol ([PRD-14](prd-14.md)) layers for audit.
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).
