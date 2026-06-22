# PRD-27: Event-Sourced, Append-Only Provenance Core with Branching (Architecture M)

> **Decide this early.** MFDB is unreleased. If the core is append-only/event-
> sourced, then PRD-12 (lifecycle) and PRD-21 (events) are **projections over the
> log**, not mutable status + bolt-on events — built once, not reworked.

## Goal

Model MFDB's provenance core as an **immutable, append-only DAG of events**
(operations and artifacts are recorded, never mutated); current state is a
**projection**; **branches/merges** are first-class. This is the logical conclusion
of MFDB's provenance-first, content-addressed design.

## Evidence (why)

- MFDB is already provenance-first (operations, edges, `derived_from`) and
  content-addressed (object store, dedup). `mfdb_branch` already gestures at
  git-like versioning.
- Mutable status columns and soft-deletes scattered across tables caused
  inconsistency (the lifecycle is a flat flag; deletes are `deleted_at` toggles).
  An append-only log gives perfect audit, reproducibility, and "what-if" branches
  for free.

## Design

### Append-only log

- One canonical, immutable **event log** (`mfdb_event`: id, ts, type, actor,
  payload, parent_event(s), branch) capturing the facts: *artifact registered*,
  *operation recorded*, *edge added*, *state transitioned*, *calibration updated*,
  *artifact superseded*. Events are never updated or hard-deleted.
- Content-addressed where it matters: an event id derives from its content +
  parents (Merkle-style), so the log is tamper-evident and naturally dedups.

### Projections (read models)

- The familiar tables (`mfdb_artifact`, `mfdb_operation`, `mfdb_edge`,
  current-state) become **projections** rebuilt/maintained from the log — fast to
  query, reconstructable from the log at any point.
- **Current state = fold over the log.** PRD-12 lifecycle states and PRD-21
  events are *the same log* viewed two ways; no separate mutable status column,
  no separate event bus table.
- "Delete" = a tombstone event, not a destructive update; history is preserved.

### Branching / merging

- A **branch** is a named pointer into the log (extends `mfdb_branch`). Work on a
  branch appends events; merging reconciles branches. Enables "what-if" reprocessing
  (try a different burst-selection parameter set on a branch) without touching the
  main line, and shareable, reproducible snapshots.

### Replayable compute specs make "what-if" concrete (Orange3 `compute_value`)

**Prior art:** Orange3 stores on each derived column a serializable transformation
that recomputes it from its source — provenance that is *replayable*, not just
recorded (see `overhaul/ORANGE3-lessons.md`). PRD-21 folds this into MFDB as a
**compute spec** on each derived artifact (operation type + `.dic`-typed parameters +
source ids).

This is what makes PRD-27 branches actionable rather than just historical: a "what-if"
branch is **"replay this artifact's compute spec with one parameter changed"**,
appended as new events on the branch. Branch + replayable spec + content-addressed
inputs = a reproducible alternative without touching the main line. PRD-27 supplies
the append-only branch; PRD-21 supplies the replay primitive; they are designed
together.

## Scope decision

This is a *foundational direction*, not a small feature. The PRD's first task is an
explicit **go/no-go** with two viable shapes:

- **Full event-sourcing** (the above): the log is the source of truth, tables are
  projections. Maximum power, larger build.
- **Append-only-lite** (fallback): keep the current tables but make provenance/
  state **append-only** (no in-place status mutation; transitions and supersessions
  are new rows; `mfdb_state_transition` from PRD-12 is the state log; deletes are
  tombstones). Most of the audit/reproducibility benefit, far smaller build, and a
  clean stepping stone to full event-sourcing later.

Recommendation: commit to **append-only-lite now** (it makes PRD-12/21 right the
first time) and keep full event-sourcing as the documented end state.

## Tasks

1. Go/no-go: pick full vs lite; record the decision and the event vocabulary.
2. Append-only invariants: provenance/state changes are new rows, never in-place
   mutation; deletes are tombstones. (Lite) or the full `mfdb_event` log + projection
   rebuild (full).
3. Make PRD-12 lifecycle and PRD-21 events *views over* the append-only record, not
   independent stores.
4. Branch pointer model (extend `mfdb_branch`); minimal branch/merge for "what-if"
   reprocessing.
5. Tests: state/provenance is reconstructable from the log; no in-place mutation of
   recorded facts; a branch diverges and re-converges; tombstone hides but preserves.

## Definition of Done

- [ ] Provenance + state are append-only (or fully event-sourced per the go/no-go);
      recorded facts are never mutated in place; deletes are tombstones.
- [ ] PRD-12 lifecycle and PRD-21 events are projections over the same record.
- [ ] Branches enable "what-if" reprocessing without touching the main line.
- [ ] State is reconstructable from the log; tests prove it.

## Definition of Clean

Append-only invariants enforced (no in-place mutation of facts); current state is a
derived projection; branches first-class; behavior-asserting tests over real chains;
delete = tombstone, never destructive.

## Relationship

Reframes **PRD-12** (lifecycle = state log projection) and **PRD-21** (events = the
log) — both must be built on this, so decide it before them. Extends `mfdb_branch`.
Sits under the operation spine (PRD-11/16): operations append events.
