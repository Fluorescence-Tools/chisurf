# MFDB / chisurf architecture optimization ideas

Grounded in concrete failures observed while implementing PRD-04/09/10 — each idea
fixes a *class* of bug, not a single instance. Ordered by leverage.

## A. One canonical identity/session context (highest leverage)

**Evidence:** "active user" is resolved two ways — settings `default_user_id`
(writes) vs the auth principal (reads). That mismatch caused the dataset browser
to show 0 ("Mine" resolved a different user than registration stamped) and forced
anonymous-fallback patches in `datasets.browse` and `datasets.open`.

**Idea:** a single `SessionContext` (current user, db handle, permissions) resolved
**once** at the boundary (GUI launch / RPC auth) and threaded explicitly. All
registration, browse, and ownership reads take it. No module-level
`_resolve_active_user_id()` consulted independently. Removes a whole bug class and
the need for "anonymous → default user" guards scattered across handlers.

## B. Dependency injection over module-global resolution

**Evidence:** `resolve_database_path` is `from … import`-bound into many module
namespaces; patching one didn't patch `api`, so setup tests read the *real* DB and
polluted each other. Tests (and my own repros) mutated `~/.chisurf/...db`.
`MFDBClient` exposed `_call` not `call`; mock clients hid it from tests.

**Ideas:**
1. Pass the DB/session **as an argument** (a request-scoped context), not via
   `resolve_database_path()` re-resolved per module. `register_result(db=…)` already
   does this; extend to every handler.
2. An **autouse test fixture** that points the settings dir + db at a temp path for
   the whole session, so no test can touch the user DB. (Real, immediate win — we
   saw tests write `rpc_test_setup` into the live DB.)
3. Define an explicit **client Protocol** (`call`, `close`) and run integration
   tests against the *real* in-process client, not a mock — the `.call`/`_call`
   bug would have been caught.

## C. Single source of truth for vocabulary + declarative migrations

**Evidence:** `operation_type` lived in **two** places — `OPERATION_TYPES` (models)
and the `bootstrap_vocabulary` dict (schema) — so `microtime_shift` was added to
one but validated against the other → silent registration failure. Migrations are
39 linear versions with `_ensure_column` scatter and local-import hacks to dodge
`UnboundLocalError`.

**Ideas:**
1. **Drive vocabulary from the `.dic` too** (it already drives tables). One seed
   source; no Python tuple/dict duplication. Operation types, artifact kinds,
   relationship types become dictionary enumerations.
2. **Declarative schema migration:** the `.dic`→DDL generator already knows the
   target schema. Compute the diff against the live schema (`introspect_sqlite_schema`)
   and auto-emit `ALTER`/`CREATE` for declared tables/columns — instead of
   hand-writing each `if version < N` block. Keep a version stamp, but the bulk of
   migrations becomes "make the DB match the dictionary." Squash the 39 historical
   versions into a baseline once stable.

## D. Provenance & lineage as a first-class API + event model

**Evidence:** lineage is ad-hoc `mfdb_edge` SQL re-written per call site (browse
group-exclusion, `derived_from`, sample joins).

**Ideas:**
1. A **lineage query API**: `ancestors(artifact)`, `descendants(artifact)`,
   `lineage_to_root`, `what_used(setup|reagent|calibration)`. One place, reused by
   browser, admin, and "if this calibration changes, which fits are affected" (the
   PRD-05 promise).
2. An **event model**: "artifact registered / operation succeeded" events on a bus.
   Enables reactive workflows (auto-trigger downstream transformers), audit, and
   the lifecycle state machine (PRD-12) without polling. chisurf already has chinet's
   reactive ports — mirror that on the data side.

## E. Workflow/pipeline engine on the transformer contract

Once PRD-11 (operation nodes) + PRD-16 (transformer contract) land, transformers
have typed input/output kinds. **Compose them into pipelines**: a node-based
workflow where output kinds of one transformer feed input kinds of the next —
the chinet dataflow graph at the *data* level, executable and recorded as a chain
of operations. Gives reproducible, shareable analysis pipelines and a visual
workflow GUI. This is the natural endgame of the operation/transformer abstraction.

## F. Thin widgets / move logic behind the API

**Evidence:** the `QComboBox` import crash shipped because logic lives in widgets
with no construction coverage; the FCS dialog wrote to MFDB on construction.

**Ideas:** widgets are pure view; all state/IO behind the api/RPC layer (the
PRD-16 split, enforced). Construction smoke tests mandatory per tool. Consider a
shared dockable-tool base so Load/Save/docks aren't re-implemented per plugin.

## G. Extract MFDB into a standalone package (already in TODO.md)

MFDB (schema, repository/API, server, admin) as a `modules/mfdb` package like
`chinet`, with a stable public API and no `chisurf` imports. Forces the clean
boundary (helps A–C), enables reuse and independent testing, and lets the schema/
generator/dictionary evolve without chisurf churn. Do it *after* the operation
spine stabilizes so the extraction freezes a good interface.

## H. Consistency / smaller wins

- **Error policy, uniform:** "MFDB unavailable → soft warn; real error → raise" is
  now in `register_result` — apply the same rule everywhere (no swallowed FK/vocab
  errors anywhere). Silent data loss was the worst class of bug this session.
- **RPC envelope contract:** one typed response shape; kill the `_call`/`_call_raw`/
  unwrap-or-not ambiguity.
- **Sample tables:** make the `mfdb_sample` (index) ↔ `flr_sample` (canonical)
  relationship explicit (a view or a single read path) so reads can't diverge again.
- **Caching:** `MmcifDictionary.load_bundled()` and the generated schema are
  recomputed often — cache them.
- **N+1 queries:** `browse_datasets` sample-count loop and per-row sample joins can
  be one query.

## Bold / breaking ideas (now possible — MFDB is unreleased)

No backward-compat burden → do the structural moves that a released system
couldn't. Each removes a root cause rather than patching symptoms.

### I. Collapse the three table families to ONE canonical model

**Evidence:** there are three parallel families — legacy `fdb_*`, flrCIF `flr_*`,
and `mfdb_*` — with reads/writes split across them. That split caused the
sample-name bug (`mfdb_sample.display_name` shadowed authoritative
`flr_sample.description`) and the dual-write in `create_sample`. **Idea:** one
canonical family rooted in flrCIF — **flrCIF (`flr_*`) stays authoritative and the
`.dic` extension dictionary extends it** where flrCIF lacks coverage (the chisurf
provenance graph + object store, declared as proper mmCIF extension categories).
Delete legacy `fdb_*`; remove the `mfdb_*` tables that *duplicate* a flrCIF concept
and repoint call sites at the authoritative `flr_*` table. Deletes the dual-table
bug class entirely and the dual-write complexity. (Biggest single simplification
available.) *Note: flrCIF is authoritative — not demoted to an export codec.*

### J. Model-driven: the `.dic` generates the *whole* data layer

The dictionary already generates DDL + validation. Push it all the way: generate
the **repository/DAO**, the admin **entity registry** (FieldSpec is already
derived), the **RPC parameter validation**, and **API docs** from the same `.dic`.
One authored spec → every layer; the hand-maintained surface (and the drift between
schema, ORM, admin, validation) largely disappears. "Dictionary dictates schema"
becomes "dictionary dictates the system."

### K. Versionless, fully declarative schema (drop the migration chain)

Unreleased ⇒ no need for 39 historical migration versions. **Reset to a baseline and
reconcile to the dictionary on open** (PRD-19, taken to its conclusion): no version
numbers for structure at all — the DB is *made to match the `.dic`*. Keep only a
tiny set of one-off **data** backfills. Removes the brittle `if version < N` chain
and the local-import/UnboundLocalError hacks for good.

### L. Repository interface + pluggable backend; split registry vs instance data

The trajectory is a shared lab LIMS (the RPC layer + server already exist).
**Design client/server-first:** an abstract repository interface with `sqlite`
(local) and `postgres` (shared) backends. And **separate two data concerns** that
are currently mixed: *registry/reference data* (the `.dic`, vocabularies, setup &
protocol **definitions** — shared, versioned, shippable) vs *instance/provenance
data* (artifacts, operations, edges — the user's experiments). They have different
lifecycles, ownership, and sync needs; separating them clarifies scoping, sharing,
and the eventual package extraction (PRD-24).

### M. Event-sourced, append-only provenance core with git-like branching

MFDB is already provenance-first and content-addressed, and `mfdb_branch` hints at
git-like versioning. **Take it to the conclusion:** the operation/artifact graph is
an **immutable, append-only DAG**; current state is a projection; branches/merges
are first-class. Gives perfect audit, reproducibility, and "what-if" branches for
free, and makes PRD-12 (lifecycle) and PRD-21 (events) natural projections over the
log rather than mutable status columns. Big direction, but it *is* the logical end
of the provenance-first design.

### N. Correctness primitives: typed IDs, units, boundary validation

- **Typed IDs.** `artifact_id`/`sample_id`/`user_id`/`operation_id` are bare
  strings — easy to swap (we saw `owner_id` confusion). `NewType`/value-object IDs
  catch the mix-up at the type level.
- **First-class units.** Parameters carry a `units` string; a real quantity/units
  system (pint-style), validated against `.dic` units, prevents unit-mismatch (the
  same class as the format dot-mismatch).
- **Validate at the boundary, not deep in a transaction.** FK/vocab errors failed
  *inside* the write transaction and were swallowed → silent loss. Validate
  requests/parameters at the RPC/registration edge (typed contracts) so failures
  are early, loud, and cheap.

## Recommendation

Top three by leverage: **(A) one identity/session context**, **(C) single-source
vocab + declarative migrations**, and **(B2) the hermetic test harness** — they
kill the bug classes that cost the most this session (owner mismatch, vocab drift,
real-DB test pollution). (D) lineage API and (E) pipeline engine are the
high-value *features* that the operation/transformer spine unlocks. (G) the package
extraction is the structural cleanup to do once the spine is stable.
