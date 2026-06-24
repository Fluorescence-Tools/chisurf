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

- [x] Lineage API answers ancestors/descendants/what_used; call sites stop writing
      bespoke edge SQL.
- [x] Event bus publishes registration/state/calibration events; at least one
      subscriber (audit or lifecycle) consumes them.
- [ ] Admin shows a provenance graph; PRD-05 impact query works. *(impact query
      primitive landed — `Lineage.impact_of`/`what_used`; the admin GUI view is the
      one remaining piece, Task 3b.)*

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

**Task 1 rollout — `Lineage` exposed on the repository.** `MFDatabase` now has a lazy
`lineage` property (`Lineage.from_connection(self.conn)`, reset on `connect()`) plus
artifact-centric accessors `get_artifact_ancestors` / `get_artifact_descendants` /
`get_artifact_impact` (PRD-05 data-side alias of descendants) /
`get_artifact_provenance_graph`, so call sites use one primitive instead of rolling
their own `mfdb_operation_artifact` traversal. _Finding:_ the graph traversal was
already consolidated into `graph.py` (`traverse_canonical_graph`) and its repository
wrappers (`get_upstream_dependencies`/`get_downstream_dependencies`/
`export_provenance_graph` → RPC `graph_upstream`/`graph_downstream`/`export_graph`);
those are **edge-rich** (they include non-operation `mfdb_edge` relationships — sample/
calibration links — and per-edge metadata) and are intentionally kept. `Lineage` is the
complementary *artifact-centric* primitive (ancestor/descendant IDs, impact, a clean
operation-graph projection); it does not replace the edge-rich path. Covered by
`test/fio/test_lineage.py` repository-accessor test.

**Task 3 (event model) — landed.** `chisurf/core/mfdb/events.py` provides a minimal
in-process `EventBus` (a process-default singleton + `publish`/`subscribe`/
`subscribe_all` helpers) and the event vocabulary (`artifact.registered`,
`operation.succeeded`, `state.changed`, `calibration.updated`). The contract holds:
publishes are **post-commit** (fired after the registration transaction in
`register_result`/`register_operation`), **best-effort and isolated** (a raising
handler is logged and swallowed — `publish` never raises, so a subscriber can never
break registration), and **synchronous/ordered** (name-specific then global
subscribers). `register_result` publishes `artifact.registered`; `register_operation`
publishes `operation.succeeded` + `artifact.registered` per output. A ready-made
`audit_log_subscriber` is provided (opt-in). Covered by `test/fio/test_events.py`
(bus delivery/ordering/isolation/unsubscribe; post-commit publish on registration; a
raising subscriber does not break registration). `state.changed`/`calibration.updated`
publish points are added with their producers (PRD-12 lifecycle / PRD-05 calibration).

**Security note (in-process only).** The MFDB event bus is an in-process Python
publish/subscribe registry — no sockets, no serialization, no RPC surface — so it has
no authentication/handshake because none is applicable (subscribers are in-process
trusted code). It is **not** bridged to the server's networked ZeroMQ PUB broadcast.
Before any networked deployment it **must not** be bridged without passing through the
authorization + payload-hygiene review in **PRD-37** (network deployment security);
treat any network-broadcast payload as readable by every authorized subscriber.

**Task 2 (embedded replayable compute spec) — landed.**
`chisurf/core/mfdb/compute_spec.py` reads a derived artifact's producing operation as a
replayable unit — **not a new format**: `ComputeSpec(operation_type, parameters,
source_artifact_ids, operation_id)` is reconstructed from the existing `mfdb_operation`
/ `mfdb_operation_artifact` / `mfdb_parameter` rows (role-indexed parameters rebuilt
into the repeatable `{value, role}` form `register_operation` accepts). `with_overrides`
is the "what-if" (new spec, no `operation_id`) behind PRD-27 branching;
`recompute(db, artifact_id)` / `replay(db, artifact_id, overrides)` dispatch to a
**pluggable replay-executor registry** (`register_replay_executor(operation_type, fn)`)
— extraction/what-if need no executor, and a missing one raises
`NoReplayExecutorError`. Exposed as `db.get_artifact_compute_spec(...)`. Covered by
`test/fio/test_compute_spec.py` (extraction incl. role-indexed round-trip, what-if
immutability, root-artifact `None`, recompute/replay dispatch + override, no-executor
error). Wiring each transformer's executor (so replay actually re-runs the pipeline) is
plugin work that fills this seam.

**Task 4 (calibration-change impact) — landed.** `Lineage.what_used` is now the full
PRD-05 "impact of change" query for *any* node, not only artifacts consumed through an
operation port. Besides the transitive operation-graph descendants it follows the
`mfdb_edge` `USAGE_RELATIONSHIPS` (`measured_sample`, `linked_to`,
`parameter_depends_on`, `uses_external_reference`; `calibrated_by` listed forward for
PRD-05, a no-op until that vocabulary term + edge writes land) to the *consumers* of a
node and adds their descendants — so a calibration/setup/reagent/sample node resolves to
the downstream artifacts it affects. `Lineage.impact_of` is the underlying primitive
(operation consumers contribute their output artifacts; artifact consumers contribute
themselves), `_edge_referrers` the one-hop reverse-edge step; `db.get_artifact_impact`
documents the broadened semantics. Backward compatible — a node with no usage edges
returns exactly its descendants. Covered by three new `test/fio/test_lineage.py` cases
(calibration reached via an artifact edge, via an operation edge, and the plain-artifact
back-compat case).

**Task 2 follow-on (replay executor) — first transformer wired.** The Micro-time
Shifter now fills the replay seam: `chisurf/plugins/tttr/tttr_microtime_shifter/api/
replay.py` self-registers a replay executor for `operation_type="microtime_shift"`. On
`recompute(db, artifact_id)` / `replay(db, artifact_id, overrides)` it materializes each
source artifact's stored TTTR blob (copied out of the content-addressed object store into
a temp file carrying the recorded `data_format` suffix so `tttrlib` can infer the
container), re-runs the conformant `MICROTIME_SHIFTER` transformer with the spec's
parameters, and registers each shifted output as a new `processed_data` artifact derived
from its source — so an artifact is now genuinely *replayable*, not just traceable. A
round-trip bug fell out and was fixed: `mfdb_parameter.value` is SQLite `REAL`, so an
int parameter (`shift=1`) reads back as `1.0` and re-validation rejected it;
`check_value_type` now accepts integral floats for `int` kinds (`1.0` ok, `1.5`
rejected). Covered by `chisurf/plugins/tttr/tttr_microtime_shifter/tests/test_replay.py`
(real-PTU recompute + replay-with-override, skipped if `tttrlib`/fixture absent;
executor registration; no-executor error). The registry tests are now order-independent
(save/restore the process-global executor slot). Burst Selection's executor (table
output, not a file) is the remaining per-transformer wiring.

**Remaining:** the admin provenance view (Task 3b, GUI-bound — needs Qt); a replay
executor for `burst_selection` (Task 2 follow-on, the same seam).

## Relationship

Builds on PRD-03/11 (operations + edges) and **PRD-27** (the event log is the
append-only core; this API projects/publishes over it). Feeds PRD-12 (lifecycle via
events) and PRD-22 (pipeline engine consumes lineage + events). Delivers PRD-05's
downstream-impact promise.
