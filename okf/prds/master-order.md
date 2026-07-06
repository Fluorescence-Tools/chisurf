---
type: Reference
title: "PRD Implementation Order"
description: The single authoritative, dependency-ordered sequence for implementing the MFDB/ChiSurf overhaul PRDs across five phases plus cross-cutting tracks.
tags: [prd, roadmap, ordering]
timestamp: '2026-07-06T00:00:00Z'
---

# PRD Implementation Order

Single authoritative sequence for the MFDB/ChiSurf overhaul: the original feature
PRDs (03–10), the operation/transformer spine + LIMS layers (11–16), and the
architecture track (17–27). Ordered by dependency → de-risking → value. Phases
ship independently; cross-cutting PRDs interleave.

## Front-loaded structural decisions

Two foundational decisions are pulled to the front because everything built after
them must be built *on* them, or it gets rebuilt. Deciding both early is the single
biggest lever for "minimize reimplementation."

- **[PRD-19](prd-19.md)** — collapse to one canonical dictionary-driven schema
  (folds ideas I + K); flrCIF stays authoritative and the `.dic` extends it;
  declarative reconcile replaces the version chain. Decide before
  [PRD-11](prd-11.md) adds many new definitions.
- **[PRD-27](prd-27.md)** — make the provenance/state core append-only (idea M).
  Decide before [PRD-12](prd-12.md) (lifecycle) and [PRD-21](prd-21.md) (events),
  which then become *projections* over it rather than mutable status + a parallel
  event store.

## Phase 0 — Current (finish first)

- **[PRD-03](prd-03.md)** result registry, **[PRD-04](prd-04.md)** burst pipeline —
  *we are here (3/4)*.
- **[PRD-09](prd-09.md)** (shifter) and **[PRD-10](prd-10.md)** (dataset browser)
  were added on this line and are largely built.

These are the *active* line, not closed-out prerequisites: PRD-04 is "implemented"
and the setup/calibration/dataset work has landed, but 03/04 remain the current
line. PRD-11–16 are **downstream** of it and slot into the existing plan rather
than treating 03/04 as finished. They reuse the PRD-03 operation/artifact tables,
the PRD-04 `.dic`→DDL generator, and the `validate_mapping` gate.

### Reconciliation with the pending original PRDs (05–08)

- **[PRD-05](prd-05.md)** calibration provenance — largely delivered by PRD-04
  sidequest C/D (g-factor reference-decay archival + time-versioned
  `mfdb_setup_calibration`); mostly-subsumed. Finish only the non-g-factor
  calibrations (gamma, crosstalk, R0) under the same pattern.
- **[PRD-07](prd-07.md)** plugin integration ("add one `register_result()` call")
  — **superseded by [PRD-16](prd-16.md)**, the strict, uniform transformer contract;
  adopt the contract instead of ad-hoc per-plugin calls.
- **[PRD-08](prd-08.md)** optical configuration — reconciled in PRD-04 (detector
  channel = base; optical = extension). Keep as an independent follow-up; orthogonal
  to PRD-11–16.
- **[PRD-06](prd-06.md)** fluorophore database — independent;
  [PRD-15](prd-15.md) reagent inventory may link to it. No conflict.

## Phase 1 — Architecture foundation (de-risk before building more)

These make everything after cheaper and safer; do them right after 03/04.

1. **[PRD-18](prd-18.md)** Dependency injection + **hermetic test harness** — land
   the autouse temp-DB fixture *immediately* (tests currently touch the real DB),
   then the DI / `MFDBClient.call` contract.
2. **[PRD-17](prd-17.md)** Canonical identity/session context — kills the
   owner-mismatch class.
3. **[PRD-19](prd-19.md)** Single canonical dictionary-driven schema (folds I + K)
   — flrCIF authoritative (extended via the `.dic`); delete legacy `fdb_*` and the
   `mfdb_*` duplicates of flrCIF concepts; vocab from the `.dic`; versionless
   declarative `reconcile_schema`. Removes the vocab-drift class and the brittle
   39-version chain *before* PRD-11 adds many new definitions. (Subsumes PRD-25 H3;
   fulfils PRD-02c by making flrCIF the single authoritative model.)
4. **[PRD-27](prd-27.md)** Append-only provenance/state core (idea M) — go/no-go +
   append-only-lite now (recorded facts never mutated in place; transitions and
   supersessions are new rows; deletes are tombstones). Decide here so PRD-12/21 are
   built once as projections.
5. **[PRD-25](prd-25.md) H1/H2 + N1/N3** (uniform fail-loud; one RPC envelope; typed
   IDs; validate at the boundary) — reinforce PRD-18; do these bits here. Remaining
   H/N items interleave later.

## Phase 2 — Operation/transformer spine + model-driven layer

6. **[PRD-11](prd-11.md)** operation-node abstraction + **[PRD-16](prd-16.md)**
   transformer contract (ship together; **supersedes [PRD-07](prd-07.md)**). Refactor
   Burst Selection (PRD-04) and Microtime Shifter (PRD-09) as the two reference
   conformant transformers, applying **[PRD-23](prd-23.md)** (thin widgets) to those
   tools as you touch them.
   - Order within the stage: land PRD-11's schema / `register_operation` first, then
     PRD-16's contract + conformance test, then refactor burst/shifter onto it.
   - Exit criteria: both transformers register via the uniform path; parameters are
     `.dic`-declared and validated; no bespoke transformer tables (retire
     `mfdb_microtime_shift`); the conformance test gates new transformers.
   - **[PRD-28](prd-28.md)** companion-tool ↔ MFDB burst-selection round trip — a
     manual-test enabler that rides this spine (send a burst selection to the
     companion tool; open one from MFDB via the dataset picker). Do once Burst
     Selection registers conformantly. The CLI handoff (`BS analyze --mfdb` →
     resolvable group artifact) is implemented and verified.
   - **[PRD-31](prd-31.md)** companion-tool headless CLI (parameter-based burst
     filtering + headless imaging) — completes the CLI leg of PRD-28; specified for
     separate implementation.
7. **[PRD-26](prd-26.md)** model-driven data layer (idea J) — the `.dic` generates
   the DAO/repository, admin entity registry, RPC validation, and docs. Builds on
   PRD-19's generator + PRD-11's parameter schemas; subsumes most hand-maintained
   admin/validation work. Do alongside or just after PRD-11.

## Phase 3 — Provenance + LIMS process layer (projections over PRD-27)

8. **[PRD-21](prd-21.md)** lineage API + event model — events feed the lifecycle;
   lineage feeds admin, browser, and PRD-05's impact query. *Projects/publishes over
   PRD-27's log.*
9. **[PRD-12](prd-12.md)** lifecycle state machine (LIMS P1) — consumes PRD-21
   events; the transition log *is* PRD-27's state projection. Highest LIMS value; the
   operation `status` folds cleanly into the generic transition log now that
   operations are uniform.
10. **[PRD-14](prd-14.md)** protocols (LIMS P3) — needs PRD-11 parameter schemas
    (`mfdb_operation_parameter_def`) and references `operation_type`; extends the same
    operation/parameter code as PRD-11, so it may run immediately after the spine if
    you prefer cohesion over the P2-before-P3 value order.
11. **[PRD-13](prd-13.md)** study/project (LIMS P2) — independent; organizational
    grouping with own+public scoping and a dataset-browser study facet; backfills
    `mfdb_study` from existing `project_id` values. May run parallel with 9/10.

Parallelization after the spine: **track A** = PRD-12 then PRD-14
(provenance/process spine); **track B** = PRD-13 then PRD-15
(organization/inventory). They touch disjoint code.

## Phase 4 — Composition + remaining features

12. **[PRD-22](prd-22.md)** pipeline/workflow engine — needs the spine (11/16) +
    lineage (21); "what-if" reprocessing rides PRD-27 branches.
13. **[PRD-15](prd-15.md)** reagent inventory (LIMS P4), **[PRD-05](prd-05.md)
    remainder** (gamma/crosstalk/R0 — g-factor done via PRD-04 C/D),
    **[PRD-06](prd-06.md)** fluorophore DB (folds in the `_dev/fluorophore_db` plugin
    as the curated real-spectra source — spectral-database/dye-vendor/photochem
    importers + curation GUI — registered into MFDB), **[PRD-08](prd-08.md)** optical
    configuration (now also folds in the Light Path Simulator as the optics
    authoring/visualization tool + a computed crosstalk/R₀ source feeding PRD-05) —
    independent; slot as needed.
- **[PRD-39](prd-39.md)** sequence provenance & external references — entity ↔
  sequence/structure database cross-refs + engineered-mutation (cysteine labeling)
  provenance via the standard `struct_ref*` categories; live cross-reference fetch +
  auto-diff. Depends on PRD-02/02a/02c; independent feature, slot as needed.

## Phase 5 — Capstone

14. **[PRD-24](prd-24.md)** extract MFDB into `modules/mfdb` — last, once
    PRD-17/18/19/26 have stabilized the interfaces.

## Cross-cutting (interleave throughout)

- **[PRD-23](prd-23.md)** thin widgets / view–api separation — apply per tool,
  starting with the transformers in Phase 2.
- **[PRD-25](prd-25.md)** consistency hardening (+ idea N) — H1/H2 + N1/N3 in
  Phase 1; H3 is subsumed by PRD-19 (one canonical schema); N2 (units) lands with
  PRD-11 params; H4 (caching), H5 (N+1), H6 (dead-code, mostly subsumed by PRD-19) as
  convenient.
- **[PRD-32](prd-32.md)** acquisition standard output folder — setup-defined default
  save path for new measurements; independent and shippable now.
- **[PRD-33](prd-33.md)** acquisition-to-MFDB registration — sample-linked or
  new-sample measurement registration; depends on PRD-02/03 and stays separate from
  the file-output path.
- **[PRD-34](prd-34.md)** BID saves to MFDB + downstream plugins ingest MFDB BIDs
  directly — when connected, a burst selection (BID) registers to MFDB (reference,
  not a loose file) and the burst tools open BIDs from the dataset picker.
  Generalizes PRD-28 to all BID producers/consumers; rides PRD-03/11/16 + PRD-10.
- **[PRD-43](prd-43.md)** Align GUI Operation History with MFDB Provenance — Phase 1
  (store history as a `project_history` artifact) is shippable now on PRD-03
  infrastructure. Phases 3–4 ride PRD-21/27.

## Dependency graph

```
PRD-03/04 (current)
   │
   ▼
Phase 1:  PRD-18 ─┬─ PRD-17 ─┬─ PRD-19 (I+K) ─┬─ PRD-27 (M)   (+ PRD-25 H1/H2 N1/N3)
                  │          │ canonical schema│ append-only core
                  ▼          ▼                 ▼
Phase 2:        PRD-11 ──► PRD-16 ──► PRD-26 (J)   (supersedes PRD-07; refactor 04+09; apply 23)
                  │        contract   model-driven layer (on 19+11)
   ┌──────────────┼───────────────┐
   ▼              ▼               ▼
PRD-21 ──► PRD-12        PRD-14        PRD-13        (Phase 3 — projections over PRD-27)
   │          │            (needs 11)   (independent)
   └────► PRD-22 (Phase 4; + spine; "what-if" on 27 branches)
                  PRD-15 / PRD-05* / PRD-06 / PRD-08  (independent features)
                  │
                  ▼
              PRD-24 (Phase 5, capstone)

Cross-cutting:  PRD-43 Phase 1 ◄── PRD-03 (shippable now)
                PRD-43 Phase 3–4 ◄── PRD-21 + PRD-27
```

## One-line rationale per architecture PRD

- **[PRD-17](prd-17.md)** identity context → one current-user resolver
  (owner-mismatch class).
- **[PRD-18](prd-18.md)** DI + hermetic tests → no test touches the real DB;
  injected db/session.
- **[PRD-19](prd-19.md)** one canonical `.dic`-driven schema (I+K) →
  three-table-family + vocab-drift + brittle-migration classes gone; flrCIF
  authoritative + `.dic`-extended; versionless reconcile.
- **[PRD-21](prd-21.md)** lineage API + events → queryable provenance, reactive
  workflows (projects over 27).
- **[PRD-22](prd-22.md)** pipeline engine → composable transformer graph (node-graph
  at data level).
- **[PRD-23](prd-23.md)** thin widgets → no logic/side-effects in GUI; smoke-tested
  tools.
- **[PRD-24](prd-24.md)** package extraction → clean `modules/mfdb` boundary
  (capstone).
- **[PRD-25](prd-25.md)** hardening (+N) → fail-loud, one RPC envelope, typed IDs,
  units, boundary validation, caching, N+1, dead code.
- **[PRD-26](prd-26.md)** model-driven layer (J) → `.dic` generates
  DAO/admin/validation/docs; the hand-maintained drift surface disappears.
- **[PRD-27](prd-27.md)** append-only provenance/state core (M) → audit,
  reproducibility, "what-if" branches; PRD-12/21 become projections, not parallel
  stores.
- **[PRD-43](prd-43.md)** GUI history alignment → preserve undo/redo history across
  DB save/restore; incrementally align with PRD-21/27 provenance.

## Recommended first action

Finish 03/04, then **Phase 1** in order: [PRD-18](prd-18.md) (test harness) →
[PRD-17](prd-17.md) (identity) → **[PRD-19](prd-19.md)** (canonical schema) →
**[PRD-27](prd-27.md)** (append-only core). Those two structural decisions (19, 27)
are what make the PRD-11/16 spine and PRD-12/21/26 build once instead of twice — the
cheapest, highest-leverage de-risking before the spine. Then start
**[PRD-11](prd-11.md) + [PRD-16](prd-16.md)** together and refactor Burst Selection
and Microtime Shifter as the two reference conformant transformers; everything else
slots onto that spine.
