# PRD-12: Lifecycle State Machines + Transition History (LIMS P1)

> **Projection over PRD-27.** If PRD-27's append-only decision is taken first (it
> should be), `mfdb_state_transition` **is** the state event log and "current
> state" is the fold over it — there is no separate mutable status flag to
> reconcile. Build this *on* PRD-27 so it is built once. The design below already
> assumes "the transition log is the source of truth; current_state is a cache";
> PRD-27 simply makes that the universal rule.

## Goal

Turn MFDB entity status from flat flags into **tracked lifecycles** with recorded
transitions (who, when, why), so a sample/dataset/operation has an auditable
"where is it and how did it get here" — the defining LIMS capability MFDB lacks.

## Background

- LIMS reference: `thirdparty/iskylims` tracks every entity through explicit state
  tables (`StatesForSample`, `RunStates`, `LibPrepareStates`, `ServiceState`, …)
  with transition history.
- MFDB today: only flat fields — `mfdb_operation.status`,
  `mfdb_artifact.validation_status`, `flr_experiment.status` — no transition
  history, no defined lifecycle, no per-entity "current state" beyond the column.
- See `overhaul/MFDB-LIMS-diagnosis.md` (P1).

## Design (generic, dictionary-driven)

One generic transition log instead of per-entity status columns:

- **`mfdb_state_transition`** (`.dic`-declared, generated, gate-covered):
  `(id PK, entity_type, entity_id, from_state, to_state, reason,
   operator_user_id FK flr_sample_users, created_at)`, index on
  `(entity_type, entity_id, created_at)`.
- **States are an extensible vocabulary** keyed by entity_type. Reuse
  `mfdb_vocabulary` with `field_name = "state:<entity_type>"` (e.g.
  `state:sample`, `state:artifact`, `state:operation`). Lifecycles:
  - **sample**: `registered → measured → processed → validated → archived`
  - **artifact/dataset**: `registered → validated → published → archived`
  - **operation**: fold the existing `pending/running/succeeded/failed/cancelled`
    into the same machinery.
- **Allowed transitions** per entity_type declared in the `.dic` (a small
  transition table seeded from the dictionary, e.g.
  `mfdb_state_transition_rule(entity_type, from_state, to_state)`), so illegal
  jumps are rejected. Optional but recommended for true LIMS behaviour.
- **Current state** = the latest non-deleted transition's `to_state`. Optionally
  cache it in a `current_state` column on the entity for fast filtering
  (dictionary-declared if added); the transition log is the source of truth.

## API

- `transition_state(entity_type, entity_id, to_state, reason="",
  operator_user_id=None)` — validates the transition against the rules, records a
  row, updates the cached `current_state` if present. Idempotent no-op if already
  in `to_state`.
- `get_state(entity_type, entity_id)` → current state.
- `get_state_history(entity_type, entity_id)` → ordered transitions.
- Registration hooks: `register_raw_measurement` sets sample/artifact initial
  state `registered`; `register_result` advances the relevant states; operation
  status changes go through `transition_state`.

## Tasks

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

## Definition of Done

- [ ] `mfdb_state_transition` exists (dict-declared, generated, gate-covered) with
      per-entity-type state vocabularies and (optional) transition rules.
- [ ] sample / artifact / operation lifecycles defined; transitions validated and
      recorded with operator + timestamp; history queryable.
- [ ] registration paths emit initial/advancing states; admin shows state+history.
- [ ] Tests pass; no flat status flag is the sole source of truth.

## Definition of Clean

`.dic` dictates the schema (no hardcoded SQL, no blob); validation rejects illegal
transitions (surfaced, not swallowed, on real errors; best-effort for
MFDB-unavailable); behavior-asserting tests; DI over monkeypatching; GUI smoke for
the admin view.

## Implementation status

**Increment 1 (schema foundation) — DONE** (`PRD-12 Increment 1` commit). The `.dic`
declares `mfdb_state_transition` (the transition log) and `mfdb_state_transition_rule`
(allowed transitions); both are `mfdb_*` extension tables so `reconcile_schema` creates
them on migrate (no hand DDL). `data/state_lifecycle_defs.json` is the authored single
source for per-entity-type states + allowed transitions (sample/artifact/operation);
`core/mfdb/lifecycle.py` (`LifecycleDef`, `load_lifecycle_defs`, `get_lifecycle_def`,
`bootstrap_lifecycle_defs`) seeds the state vocabularies (`mfdb_vocabulary`
`field_name="state:<entity_type>"`) and the rule table (idempotent), wired into both
migrate paths in `schema.py`. Covered by `test/fio/test_lifecycle_schema.py` (6).

**Increment 2 (transition API) — DONE** (`PRD-12 Increment 2` commit). `repository.py`
now has `transition_state(entity_type, entity_id, to_state, reason="",
operator_user_id=None)` (validates against `mfdb_state_transition_rule`, raises
`lifecycle.StateTransitionError` on an illegal jump, idempotent no-op returning `False`
when already in `to_state`, records the row + audit log, publishes PRD-21's
`state.changed` post-commit), `get_state` (latest non-deleted `to_state`), and
`get_state_history` (ordered). `transition_id` is assigned MAX+1. Covered by
`test/fio/test_lifecycle.py` (9): initial/advancing, illegal jump + illegal initial
rejected (state unchanged), idempotent no-op (no row, no event), ordered history,
branching operation lifecycle, `state.changed` payload, audit log.

### ▶ START NEXT — Increment 3 (wire registration paths; higher blast radius)
Emit lifecycle transitions from the registration paths in
`chisurf/core/mfdb/result_registry.py`, **best-effort** (wrap in try/except, never break
registration — same posture as the event publish there):
- `register_raw_measurement` → set the raw artifact's initial state `registered` (and,
  when a `sample_id` is linked, the sample's `registered` if it has no state yet).
- `register_result` → advance the output artifact to `registered` (its lifecycle start);
  leave richer advancement (`validated`/`published`) to explicit callers/admin for now.
- Operation status: `register_operation`/`register_result` already record
  `status="succeeded"`; optionally mirror that through `transition_state("operation", …)`
  so the operation lifecycle has history (pending→running→succeeded). Keep it best-effort.
Tests: after `register_raw_measurement`/`register_result`, `db.get_state("artifact", id)`
== `registered`; a `state.changed` subscriber fires. Watch the existing registration
test suites for regressions (run the `test/fio` mfdb slice in `arm64`, `-o addopts=""`).
Then **Increment 4** — mfdb-admin state+history view (Qt; arm64 has PyQt5): show current
state + `get_state_history` per entity and allow admin transitions.

## Original recipe (full reference)

**Increment 1 — schema foundation (committable on its own; verify with the gate).**
1. `.dic` (`chisurf/core/mfdb/data/mfdb_flr_ext.dic`): add a `save_mfdb_state_transition`
   category block + one `save__mfdb_state_transition.<col>` block per column. Use the
   `save_mfdb_operation_parameter_def` block (~line 3242) as the exact template — each
   column block needs `_item.name`, `_item.category_id`, `_item_type.code`
   (`int`/`code`/`text`/`boolean`), `_item.mandatory_code`, and the
   `_chisurf_schema.table_name`/`column_name` pair. Columns: `transition_id` (int PK),
   `entity_type` (code), `entity_id` (code), `from_state` (code, optional),
   `to_state` (code), `reason` (text, optional), `operator_user_id` (code, optional),
   `created_at`/`updated_at`/`deleted_at` (text). Do the same for the small
   `mfdb_state_transition_rule` (`entity_type`, `from_state`, `to_state`).
2. DDL: add `CREATE TABLE IF NOT EXISTS mfdb_state_transition (...)` and
   `mfdb_state_transition_rule (...)` to `CREATE_TABLES_SQL` in
   `chisurf/core/mfdb/schema.py` (~line 70); add indices to `CREATE_INDICES_SQL`
   (~line 885): `idx_mfdb_state_transition_entity (entity_type, entity_id, created_at)`
   + a `deleted_at` index. `FRESH_DB_TABLES_SQL`/`FRESH_DB_INDICES_SQL` derive
   automatically.
3. Seed per-entity-type state vocabularies into `mfdb_vocabulary`
   (`field_name = "state:sample"|"state:artifact"|"state:operation"`) and the
   transition rules — follow the existing vocabulary seed path; the lifecycles are
   sample `registered→measured→processed→validated→archived`, artifact
   `registered→validated→published→archived`, operation
   `pending→running→succeeded→failed→cancelled`.
4. Verify: run the schema/dictionary gate test (live⊇declared + vocab==dictionary) in
   the `arm64` env, `-o addopts=""`.

**Increment 2 — repository API + tests.** `transition_state(entity_type, entity_id,
to_state, reason="", operator_user_id=None)` (validate against the rule table; idempotent
no-op if already in `to_state`; record a row; publish PRD-21's `EVENT_STATE_CHANGED` =
`state.changed` post-commit — the constant already exists in `events.py`, just add the
publish point here), `get_state` (latest non-deleted `to_state`), `get_state_history`
(ordered). Behaviour tests: legal recorded, illegal rejected, history ordered, current
resolves, idempotent re-transition.

**Increment 3 (higher blast radius) — wire registration paths** (`register_raw_measurement`
initial `registered`; `register_result` advances) best-effort; **Increment 4** — the
mfdb-admin state+history view (Qt; arm64 has PyQt5).

## Relationship

A **projection over PRD-27** (the append-only core): the transition log is the
state event stream; current state is the fold. Shares the `mfdb_vocabulary` /
dictionary machinery with PRD-19; emitted via PRD-21's event model. Decide PRD-27
first so the status model is not built twice.
