---
type: Reference
title: MFDB Architecture Ideas
description: The durable design rationale behind the MFDB overhaul PRDs — each decision grounded in an observed bug class.
tags: [reference, mfdb, architecture]
timestamp: '2026-07-06T00:00:00Z'
---

# Reference

This concept captures the design rationale behind the MFDB overhaul PRDs. Each
idea was grounded in a concrete failure observed while building out the metadata /
provenance store — it fixes a *class* of bug, not a single instance. The problems
are recorded here so the "why" survives even as the implementation and the
individual PRDs evolve. Ideas are ordered by leverage; each links to the PRD that
carries it forward.

## Canonical identity / session context

**Observed problem.** The "active user" was resolved two different ways —
settings `default_user_id` on writes vs. the auth principal on reads. That mismatch
made the dataset browser show 0 results ("Mine" resolved a different user than
registration had stamped) and forced scattered anonymous-fallback patches in
`datasets.browse` and `datasets.open`.

**Decision.** Resolve a single `SessionContext` (current user, db handle,
permissions) **once** at the boundary — GUI launch or RPC auth — and thread it
explicitly through registration, browse, and ownership reads. No module-level
`_resolve_active_user_id()` consulted independently. This removes a whole bug
class and the need for "anonymous → default user" guards across handlers.
Implemented by [PRD-17](/prds/prd-17.md).

## Dependency injection over module-global resolution + a hermetic test harness

**Observed problem.** `resolve_database_path` was `from … import`-bound into many
module namespaces; patching one did not patch `api`, so setup tests read the
*real* database and polluted each other — tests wrote `rpc_test_setup` into the
live user db. The client exposed `_call` but not `call`, so mock clients hid that
mismatch from tests.

**Decisions.**
1. Pass the db/session **as an argument** (a request-scoped context) rather than
   re-resolving `resolve_database_path()` per module. `register_result(db=…)`
   already did this; extend it to every handler.
2. Add an **autouse test fixture** that points the settings dir + db at a temp
   path for the whole session, so no test can touch the user db.
3. Define an explicit **client Protocol** (`call`, `close`) and run integration
   tests against the *real* in-process client, not a mock — the `.call`/`_call`
   bug would have been caught.

These correctness/consistency and boundary-validation concerns are carried by
[PRD-25](/prds/prd-25.md).

## Single source of truth for vocabulary + declarative migrations

**Observed problem.** `operation_type` lived in **two** places —
`OPERATION_TYPES` in the models and the `bootstrap_vocabulary` dict in the schema —
so `microtime_shift` was added to one but validated against the other, producing a
silent registration failure. Migrations had grown to 39 linear versions with
`_ensure_column` scatter and local-import hacks to dodge `UnboundLocalError`.

**Decisions.**
1. **Drive vocabulary from the `.dic` dictionary too** (it already drives the
   tables): one seed source, no Python tuple/dict duplication. Operation types,
   artifact kinds, and relationship types become dictionary enumerations.
2. **Declarative schema migration:** the dictionary→DDL generator already knows
   the target schema. Compute the diff against the introspected live schema and
   auto-emit `ALTER`/`CREATE` for declared tables/columns, instead of
   hand-writing each `if version < N` block. Keep a version stamp, but the bulk of
   migration becomes "make the db match the dictionary." Squash the historical
   versions into a baseline once stable.

Implemented by [PRD-19](/prds/prd-19.md).

## Provenance & lineage as a first-class API + event model

**Observed problem.** Lineage was ad-hoc `mfdb_edge` SQL re-written per call site
(browse group-exclusion, `derived_from`, sample joins).

**Decisions.**
1. A **lineage query API**: `ancestors(artifact)`, `descendants(artifact)`,
   `lineage_to_root`, `what_used(setup|reagent|calibration)`. One place, reused by
   the browser, admin, and the "if this calibration changes, which fits are
   affected" question.
2. An **event model**: "artifact registered / operation succeeded" events on a
   bus. This enables reactive workflows (auto-triggering downstream transformers),
   audit, and a lifecycle state machine without polling. chisurf already has
   chinet's reactive ports on the compute side — mirror that on the data side.

Implemented by [PRD-21](/prds/prd-21.md).

## Workflow / pipeline engine on the transformer contract

Once operation nodes and a typed transformer contract exist, transformers have
typed input/output kinds. **Compose them into pipelines:** a node-based workflow
where the output kinds of one transformer feed the input kinds of the next — the
chinet dataflow graph at the *data* level, executable and recorded as a chain of
operations. This yields reproducible, shareable analysis pipelines and a visual
workflow GUI, the natural endgame of the operation/transformer abstraction.
Implemented by [PRD-22](/prds/prd-22.md).

## Thin widgets / logic behind the API

**Observed problem.** A `QComboBox` import crash shipped because logic lived in
widgets with no construction coverage; an FCS dialog wrote to MFDB on
construction.

**Decision.** Widgets are pure view; all state and IO live behind the api/RPC
layer, enforced. Construction smoke tests are mandatory per tool. A shared
dockable-tool base avoids re-implementing Load/Save/docks per plugin.

## Extract MFDB into a standalone package

MFDB (schema, repository/API, server, admin) should become an independent module
like chinet, with a stable public API and no `chisurf` imports. That forces the
clean boundary (which helps identity, DI, and vocabulary above), enables reuse and
independent testing, and lets the schema/generator/dictionary evolve without
chisurf churn. Do it *after* the operation spine stabilizes, so the extraction
freezes a good interface. Implemented by [PRD-24](/prds/prd-24.md).

## Consistency / smaller wins

- **Uniform error policy:** "MFDB unavailable → soft warn; real error → raise"
  should apply everywhere, with no swallowed FK/vocab errors. Silent data loss was
  the worst bug class observed.
- **RPC envelope contract:** one typed response shape, killing the
  `_call`/`_call_raw`/unwrap-or-not ambiguity.
- **Sample tables:** make the `mfdb_sample` (index) ↔ `flr_sample` (canonical)
  relationship explicit (a view or a single read path) so reads can't diverge.
- **Caching:** the bundled dictionary and generated schema are recomputed often —
  cache them.
- **N+1 queries:** the `browse_datasets` sample-count loop and per-row sample
  joins can each be a single query.

These are carried by [PRD-25](/prds/prd-25.md).

## Bold / breaking ideas (possible because MFDB is unreleased)

With no backward-compat burden, structural moves a released system couldn't make
become available. Each removes a root cause rather than patching a symptom.

### Collapse the three table families to ONE canonical model

**Observed problem.** Three parallel table families coexisted — a legacy `fdb_*`
family, the flrCIF `flr_*` family, and the `mfdb_*` family — with reads and writes
split across them. That split caused a sample-name bug
(`mfdb_sample.display_name` shadowed the authoritative `flr_sample.description`)
and a dual-write in `create_sample`.

**Decision.** One canonical family rooted in flrCIF: **`flr_*` stays authoritative
and the `.dic` extension dictionary extends it** where flrCIF lacks coverage (the
chisurf provenance graph + object store, declared as proper mmCIF extension
categories). Delete the legacy `fdb_*` family; remove the `mfdb_*` tables that
*duplicate* a flrCIF concept and repoint call sites at the authoritative `flr_*`
table. This deletes the dual-table bug class and the dual-write complexity — the
biggest single simplification available. flrCIF stays authoritative; it is not
demoted to an export codec. Part of [PRD-19](/prds/prd-19.md).

### Model-driven: the `.dic` generates the *whole* data layer

The dictionary already generates DDL + validation. Push it all the way: generate
the **repository/DAO**, the admin **entity registry** (its FieldSpec is already
derived), the **RPC parameter validation**, and **API docs** from the same `.dic`.
One authored spec feeds every layer, so the hand-maintained surface — and the
drift between schema, ORM, admin, and validation — largely disappears.
"Dictionary dictates schema" becomes "dictionary dictates the system." Part of
[PRD-19](/prds/prd-19.md).

### Versionless, fully declarative schema (drop the migration chain)

Because MFDB is unreleased, the historical migration versions carry no value.
**Reset to a baseline and reconcile to the dictionary on open**, taking the
declarative approach to its conclusion: no version numbers for structure at all —
the db is *made to match the `.dic`*. Keep only a tiny set of one-off **data**
backfills. This removes the brittle `if version < N` chain and the
local-import/`UnboundLocalError` hacks. Part of [PRD-19](/prds/prd-19.md).

### Repository interface + pluggable backend; split registry vs. instance data

The trajectory is a shared-lab metadata store (the RPC layer + server already
exist). **Design client/server-first:** an abstract repository interface with an
embedded file-based backend (local) and a shared server-grade relational backend
(multi-user). And **separate two data concerns** currently mixed together:
*registry/reference data* (the `.dic`, vocabularies, setup & protocol
**definitions** — shared, versioned, shippable) vs. *instance/provenance data*
(artifacts, operations, edges — the user's experiments). They have different
lifecycles, ownership, and sync needs; separating them clarifies scoping, sharing,
and the eventual package extraction. Part of [PRD-24](/prds/prd-24.md).

### Event-sourced, append-only provenance core with version-control-style branching

MFDB is already provenance-first and content-addressed, and an `mfdb_branch` hint
points at version-control-style versioning. **Take it to the conclusion:** the
operation/artifact graph is an **immutable, append-only DAG**; current state is a
projection; branches/merges are first-class. This gives perfect audit,
reproducibility, and "what-if" branches for free, and makes lifecycle and events
natural projections over the log rather than mutable status columns. A large
direction, but it is the logical end of the provenance-first design. Implemented
by [PRD-27](/prds/prd-27.md), with events feeding [PRD-21](/prds/prd-21.md).

### Correctness primitives: typed IDs, units, boundary validation

- **Typed IDs.** `artifact_id` / `sample_id` / `user_id` / `operation_id` were
  bare strings, easy to swap (an `owner_id` confusion was observed). Value-object /
  `NewType` IDs catch the mix-up at the type level.
- **First-class units.** Parameters carry a `units` string; a real quantity/units
  system, validated against `.dic` units, prevents unit-mismatch bugs (the same
  class as a format dot-mismatch).
- **Validate at the boundary, not deep in a transaction.** FK/vocab errors failed
  *inside* the write transaction and were swallowed, causing silent loss. Validate
  requests/parameters at the RPC/registration edge with typed contracts so
  failures are early, loud, and cheap.

Implemented by [PRD-25](/prds/prd-25.md).

## Highest-leverage summary

The top three by leverage were the ones that killed the bug classes that cost the
most: one identity/session context ([PRD-17](/prds/prd-17.md)), single-source
vocabulary + declarative migrations ([PRD-19](/prds/prd-19.md)), and the hermetic
test harness ([PRD-25](/prds/prd-25.md)) — owner mismatch, vocabulary drift, and
real-db test pollution respectively. The lineage API
([PRD-21](/prds/prd-21.md)) and the pipeline engine
([PRD-22](/prds/prd-22.md)) are the high-value *features* that the
operation/transformer spine unlocks; the package extraction
([PRD-24](/prds/prd-24.md)) is the structural cleanup to do once that spine is
stable.
