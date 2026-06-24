# PRD-21: Provenance/Lineage Query API + Event Model (Architecture D)

> **Projection over PRD-27.** If the core is append-only/event-sourced (PRD-27),
> the "event model" here is **the log itself**, and the lineage API is a read
> projection over the recorded events — not a separate event-bus table bolted on.
> Build on PRD-27 so events are stored once (the log) and merely *published* to
> subscribers, rather than maintained in a parallel store.

## Goal

Make the provenance graph queryable through a first-class **lineage API** instead
of ad-hoc `mfdb_edge` SQL, and add an **event model** so registration/state changes
can drive reactive behaviour (downstream triggers, lifecycle, audit).

## Evidence (why)

Lineage is hand-written `mfdb_edge` SQL at each call site (browse group-exclusion,
`derived_from`, sample joins, calibration links). PRD-05's promise — "when a
calibration value changes, all downstream fits that used it can be identified" —
has no query primitive behind it.

## Design

### Lineage API (read)

A `Lineage` service over `mfdb_edge` / `mfdb_operation_artifact`:

- `ancestors(artifact_id)`, `descendants(artifact_id)`, `lineage_to_root(artifact_id)`
  (full derivation chain).
- `what_used(node)` / `what_was_produced_from(node)` for setups, calibrations,
  reagents, protocols (e.g. "which fits used calibration C / setup S / lot L").
- `provenance_graph(node, depth)` → nodes+edges for visualization (mfdb-admin /
  the future pipeline GUI).

Reused by the dataset browser, admin lineage view, and the PRD-05
"impact-of-change" query.

### Embedded, replayable provenance (Orange3 `compute_value`) — fold

**Prior art:** Orange3 bakes provenance into its data model. Every derived column
(`Variable`) carries a serializable `compute_value` — a `Transformation` object that
knows *how to recompute this column from its source* (`var.copy(compute_value=
ReplaceUnknowns(var, value))`). Lineage is intrinsic and **recomputable**, not a
side-table afterthought; the transform is picklable and re-applies to new data.
See `overhaul/ORANGE3-lessons.md`.

**Apply to MFDB:** alongside the `mfdb_edge`/`mfdb_operation` records, store on each
derived artifact a **compact, serializable compute spec** — `(operation_type,
parameter set, source_artifact_ids)` — i.e. the PRD-11 operation node captured as a
replayable value (its `.dic`-typed parameters are exactly the spec schema). Then:

- An artifact is not just *traceable* but *replayable*: "recompute this artifact" =
  re-run its compute spec; "what-if" = re-run with one parameter changed (this is the
  concrete mechanism behind **PRD-27** branches).
- The lineage API reads the embedded specs; `mfdb_edge` becomes a derived **index/
  projection** of them, not the source of truth (consistent with PRD-27's log-as-
  truth direction).
- Reproducibility falls out for free: a recorded spec + content-addressed inputs
  reproduces the output.

Keep it a thin record (reuse PRD-11's operation + parameter rows; the "spec" is just
those rows addressed as a unit), not a second serialization format.

### Event model (write-side)

- An in-process event bus: `artifact.registered`, `operation.succeeded`,
  `state.changed` (PRD-12), `calibration.updated`. Publishers are the registration/
  transition paths; subscribers are optional.
- Enables: reactive workflows (auto-run a downstream transformer when its input
  appears), audit logging, lifecycle advancement, and cache invalidation — without
  polling. Mirrors chinet's reactive ports on the data side.
- Synchronous, best-effort, ordered; never breaks the registering transaction
  (handlers run after commit).

## Tasks

1. `Lineage` service with the queries above; replace ad-hoc edge SQL in browse/
   admin with it.
2. Embedded compute spec: address each derived artifact's PRD-11 operation +
   parameter rows as a replayable unit; a `recompute(artifact_id)` /
   `replay(artifact_id, parameter_overrides)` primitive (the basis for PRD-27
   "what-if"). `mfdb_edge` becomes a projection of these.
3. Event bus + publish points in registration / state transitions; document the
   event vocabulary.
3. mfdb-admin: a lineage/provenance view for an artifact (ancestors → node →
   descendants).
4. Wire PRD-05's "impact of calibration change" to `what_used`.
5. Tests: lineage over a built chain (sample → raw → shift → burst); `what_used`
   for a calibration; an event fires on registration and a subscriber runs.

## Definition of Done

- [ ] Lineage API answers ancestors/descendants/what_used; call sites stop writing
      bespoke edge SQL.
- [ ] Event bus publishes registration/state/calibration events; at least one
      subscriber (audit or lifecycle) consumes them.
- [ ] Admin shows a provenance graph; PRD-05 impact query works.

## Definition of Clean

One lineage API (no duplicated edge SQL); events are post-commit, best-effort,
never break registration; behavior-asserting tests over real chains.

## Implementation status

**Task 1 (lineage read API) — core landed.** `chisurf/core/mfdb/lineage.py` provides
`Lineage` (`from_db`/`from_connection`), a read-only service over the authoritative
operation graph (`mfdb_operation_artifact`): an artifact is *produced by* the
operations listing it as `output` and *consumed by* those listing it as `input`, and
ancestry/descent is the transitive closure over artifact→operation→artifact hops. It
deliberately does **not** rely on the dual-written `derived_from` `mfdb_edge` rows
(direction-ambiguous), so results are correct by construction. API: `ancestors`,
`descendants`, `lineage_to_root`, `parents`/`children` (one-hop), `what_used` (the
data-side of PRD-05 downstream-impact = transitive descendants), and
`provenance_graph(artifact_id, depth)` → `{"nodes", "edges"}` (artifact + operation
nodes, `produced`/`input_to` edges) for the admin/pipeline visualization. Covered by
`test/fio/test_lineage.py` (7 tests over a real `sample → raw → microtime_shift →
burst_selection` chain: ancestors/descendants order, one-hop parents/children,
`lineage_to_root`, `what_used` impact, the graph projection, and the isolated-artifact
empty case).

**Remaining:** replace the ad-hoc edge SQL in browse/admin with `Lineage` (Task 1
rollout); embedded replayable compute spec + `recompute`/`replay` (Task 2); the
post-commit event bus + publish points (Task 3); the admin provenance view (Task 3b);
wire PRD-05's calibration-change impact to `what_used` for non-artifact (setup/
calibration/reagent) nodes (Task 4).

## Relationship

Builds on PRD-03/11 (operations + edges) and **PRD-27** (the event log is the
append-only core; this API projects/publishes over it). Feeds PRD-12 (lifecycle via
events) and PRD-22 (pipeline engine consumes lineage + events). Delivers PRD-05's
downstream-impact promise.
