# PRD-25: Consistency Hardening + Correctness Primitives (Architecture H + N)

> **Folds in** idea **N** (typed IDs, first-class units, boundary validation).

## Goal

A set of cross-cutting correctness/consistency changes that each remove a latent
bug class. The H items are small/incremental; the N items (N1–N3) are correctness
*primitives* that prevent whole classes at the type/boundary level.

## Items

### H1 — Uniform fail-loud error policy

`register_result` now distinguishes "MFDB unavailable → soft warn" from "real error
→ raise" (after the silent FK/vocab data-loss bugs). **Apply the same rule
everywhere**: every MFDB write/read path warns only for "no DB", and surfaces
genuine errors (FK, vocab, integrity, transport). No `except Exception: pass` over
real failures. Audit `_call_rpc` and handlers (the swallowed `AttributeError` that
showed "0 datasets").

### H2 — One RPC envelope contract

RPC responses are inconsistent (`_call` vs `_call_raw`, wrapped vs unwrapped
`{"ok","result"}`). Define **one** response shape + one client-side unwrap, typed.
Handlers return data; the dispatcher wraps; the client unwraps once. Kills the
"reads the wrong level → empty" class.

### H3 — Explicit `mfdb_sample` ↔ `flr_sample` relationship

Two sample tables (`mfdb_sample` index, `flr_sample` flrCIF-canonical) had reads
split across them → nameless/missing samples. Make the relationship explicit: a
single canonical read path (or a SQL view `sample_v` joining/coalescing), so reads
cannot diverge again. (We fixed the symptom; this fixes the structure.)

### H4 — Caching

`MmcifDictionary.load_bundled()` and the generated schema are recomputed
repeatedly (every migrate/open, multiple handlers). Cache the parsed dictionary and
generated DDL at process scope (invalidate on `.dic` change in dev).

### H5 — Kill N+1 queries

`browse_datasets` does a per-row sample join + a separate sample-count loop. Fold
into single queries. Audit other list endpoints for N+1.

### H6 — Drop dead/legacy paths

`fdb_*` and legacy duplicate tables/readers linger (README: "new code must only use
`mfdb_*`"). Remove dead code paths as they're confirmed unused (per the overhaul
ground rule "delete dead code"). *(Largely subsumed by PRD-19's collapse to one
canonical schema.)*

### N1 — Typed IDs

`artifact_id` / `sample_id` / `user_id` / `operation_id` / `setup_id` are bare
strings — trivially swapped (we hit `owner_id` confusion). Introduce
`NewType`/value-object IDs (`ArtifactId`, `SampleId`, `UserId`, …) so the type
checker catches the mix-up. Apply at the repository/API boundary.

### N2 — First-class units / quantities

Parameters carry a free-text `units` column. Add a real quantity/units layer
(pint-style) for parameter values, validated against the `.dic` `_item_units.code`,
so unit-mismatch (the same class as the data_format dot-mismatch) is impossible.
Operation parameters (PRD-11) and setup/calibration values are the first consumers.

### N3 — Validate at the boundary, not deep in a transaction

FK/vocab errors failed *inside* the write transaction and were swallowed → silent
data loss. Validate requests + parameters at the **RPC/registration edge** (typed
contracts, dictionary-driven via PRD-26) so failures are early, loud, and cheap —
before opening a transaction. (Merges with H1; H1 is the policy, N3 is moving the
check to the edge.)

## Tasks

One focused change + test per item (H1–H6); they are independent and can land in
any order, interleaved with other PRDs.

## Definition of Done

- [ ] H1 error policy applied across MFDB read/write/RPC; no real error swallowed.
- [ ] H2 single typed RPC envelope + one unwrap.
- [ ] H3 single canonical sample read path / view.
- [ ] H4 dictionary + generated schema cached.
- [ ] H5 browse/list endpoints free of N+1.
- [ ] H6 confirmed-dead `fdb_*`/legacy paths removed.

## Definition of Clean

Fail-loud on real errors; one RPC contract; no duplicated read paths; behavior-
asserting tests per item; delete dead code (don't alias).

## Relationship

Cross-cutting; H1/H2 reinforce PRD-18; H3 finishes the sample-table reconciliation
started in the flrCIF-canonical fix; the rest are independent hygiene.
