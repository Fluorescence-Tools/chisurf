---
type: PRD
prd: "25"
title: "PRD-25: Consistency Hardening + Correctness Primitives"
description: A set of cross-cutting correctness changes — uniform fail-loud errors, one RPC envelope, a single sample read path, caching, N+1 removal, dead-code removal — plus typed IDs, first-class units, and boundary validation.
status: in-progress
phase: "1"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-25 bundles cross-cutting correctness and consistency changes that each remove a latent bug class. The hardening items apply a uniform fail-loud error policy (warn only for a missing database, surface genuine FK/vocab/integrity/transport errors), define one typed RPC envelope with a single client-side unwrap, provide a single canonical sample read path, cache the parsed dictionary and generated DDL, remove N+1 queries from list endpoints, and drop confirmed-dead legacy paths. The correctness primitives add typed IDs (distinct value-object types for artifact, sample, user, operation, and setup ids), a first-class units/quantities layer validated against the dictionary, and validation moved to the RPC/registration boundary rather than deep inside a transaction.

# Status
In-progress (phase 1, STATUS TABLE authoritative). Items are independent and land in any order, interleaved with other PRDs.

# Goal
A set of cross-cutting correctness/consistency changes that each remove a latent bug class. The H items are small/incremental; the N items (N1–N3) are correctness *primitives* that prevent whole classes at the type/boundary level. This PRD folds in the "typed IDs, first-class units, boundary validation" idea.

# Items

## H1 — Uniform fail-loud error policy
`register_result` distinguishes "MFDB unavailable → soft warn" from "real error → raise" (after the silent FK/vocab data-loss bugs). **Apply the same rule everywhere**: every MFDB write/read path warns only for "no DB", and surfaces genuine errors (FK, vocab, integrity, transport). No `except Exception: pass` over real failures. Audit `_call_rpc` and handlers (the swallowed `AttributeError` that showed "0 datasets").

## H2 — One RPC envelope contract
RPC responses are inconsistent (`_call` vs `_call_raw`, wrapped vs unwrapped `{"ok","result"}`). Define **one** response shape + one client-side unwrap, typed. Handlers return data; the dispatcher wraps; the client unwraps once. Kills the "reads the wrong level → empty" class.

## H3 — Explicit `mfdb_sample` ↔ `flr_sample` relationship
Two sample tables (`mfdb_sample` index, `flr_sample` flrCIF-canonical) had reads split across them → nameless/missing samples. Make the relationship explicit: a single canonical read path (or a SQL view `sample_v` joining/coalescing), so reads cannot diverge again. (The symptom was fixed; this fixes the structure.)

## H4 — Caching
`MmcifDictionary.load_bundled()` and the generated schema are recomputed repeatedly (every migrate/open, multiple handlers). Cache the parsed dictionary and generated DDL at process scope (invalidate on `.dic` change in dev).

## H5 — Kill N+1 queries
`browse_datasets` does a per-row sample join + a separate sample-count loop. Fold into single queries. Audit other list endpoints for N+1.

## H6 — Drop dead/legacy paths
`fdb_*` and legacy duplicate tables/readers linger (README: "new code must only use `mfdb_*`"). Remove dead code paths as they're confirmed unused (per the overhaul ground rule "delete dead code"). *(Largely subsumed by [PRD-19](prd-19.md)'s collapse to one canonical schema.)*

## N1 — Typed IDs
`artifact_id` / `sample_id` / `user_id` / `operation_id` / `setup_id` are bare strings — trivially swapped (the `owner_id` confusion). Introduce `NewType`/value-object IDs (`ArtifactId`, `SampleId`, `UserId`, …) so the type checker catches the mix-up. Apply at the repository/API boundary.

## N2 — First-class units / quantities
Parameters carry a free-text `units` column. Add a real quantity/units layer (dimension-checked) for parameter values, validated against the `.dic` `_item_units.code`, so unit-mismatch (the same class as the data_format dot-mismatch) is impossible. Operation parameters ([PRD-11](prd-11.md)) and setup/calibration values are the first consumers.

## N3 — Validate at the boundary, not deep in a transaction
FK/vocab errors failed *inside* the write transaction and were swallowed → silent data loss. Validate requests + parameters at the **RPC/registration edge** (typed contracts, dictionary-driven via [PRD-26](prd-26.md)) so failures are early, loud, and cheap — before opening a transaction. (Merges with H1; H1 is the policy, N3 is moving the check to the edge.)

# Tasks
One focused change + test per item (H1–H6); they are independent and can land in any order, interleaved with other PRDs.

# Definition of Done
- [ ] H1 error policy applied across MFDB read/write/RPC; no real error swallowed.
- [ ] H2 single typed RPC envelope + one unwrap.
- [ ] H3 single canonical sample read path / view.
- [ ] H4 dictionary + generated schema cached.
- [ ] H5 browse/list endpoints free of N+1.
- [ ] H6 confirmed-dead `fdb_*`/legacy paths removed.

# Definition of Clean
Fail-loud on real errors; one RPC contract; no duplicated read paths; behavior-asserting tests per item; delete dead code (don't alias).

# Relationships
- Cross-cutting; the error-policy and RPC-envelope items (H1/H2) reinforce [PRD-18](prd-18.md).
- The sample-table item (H3) finishes the reconciliation completed by [PRD-19](prd-19.md) (which subsumes it) and depends on the flrCIF-canonical fix.
- Boundary validation (N3) is dictionary-driven via [PRD-26](prd-26.md), reusing the operation parameter schemas of [PRD-11](prd-11.md).
- Targets the [MFDB target](/specs/mfdb.md) and the [RPC target](/specs/rpc.md).
