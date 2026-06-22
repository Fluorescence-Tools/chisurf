# Implementation order — operation spine + LIMS layers (PRD-11–16)

Sequenced by dependency → value → cohesion. Each stage is shippable on its own and
de-risks the next. All stages follow the established Definition of Clean
(`.dic`-dictates-schema, generated DDL, total-coverage gate, fail-loud on real
errors, DI over monkeypatching, GUI smoke).

## Where we are now (reconciled with the existing plan)

Per `overhaul/README.md` the master sequence is PRD-01 → 02x ✓ → **03 → 04** →
05–08. **The team is currently on PRD-03 (result registry) and PRD-04 (burst
pipeline)** — "3/4". These are substantially built in the codebase (PRD-04 says
"implemented"; this conversation landed the setup/calibration/dataset work), but
they are the *active* line, not closed-out prerequisites. PRD-09 (shifter) and
PRD-10 (dataset browser) were added on top during this work and are largely
implemented but are **not yet in the README index**.

So PRD-11–16 are **downstream of the current 03/04 line** and should be slotted
into the existing plan, not treated as if 03/04 were finished. They reuse the
PRD-03 operation/artifact tables, the PRD-04 `.dic`→DDL generator, and the
validate_mapping gate.

### Reconciliation with the pending original PRDs (05–08)

| Existing PRD | Overlap with the new PRDs | Resolution |
|---|---|---|
| **PRD-05** calibration provenance | Largely delivered by **PRD-04 sidequest C/D** (g-factor reference-decay archival + time-versioned `mfdb_setup_calibration`) | Mark PRD-05 mostly-subsumed; finish only the non-g-factor calibrations (gamma, crosstalk, R0) under the same pattern |
| **PRD-07** plugin integration ("add one `register_result()` call") | **PRD-16** is the strict, uniform version of this | **PRD-16 supersedes PRD-07** — adopt the transformer contract instead of ad-hoc per-plugin calls |
| **PRD-08** optical configuration | Reconciled in PRD-04 (detector channel = base; optical = extension) | Keep PRD-08 as an independent follow-up; it is orthogonal to PRD-11–16 |
| **PRD-06** fluorophore database | Independent; PRD-15 reagent inventory may link to it | No conflict |

So the effective forward order is: **finish 03/04 → (optionally 08, 06 independently) → Stage 1 (PRD-11+16, which subsumes 07) → Stage 2–5 (LIMS) → mop up PRD-05 remainder**.

## Stage 1 — Operation/Transformer spine (do together) ★ start here

**PRD-11** (data-operation node + `.dic` parameter schemas + role-indexed
parameters; retire `mfdb_microtime_shift`) **and PRD-16** (transformer contract;
refactor Burst Selection + Microtime Shifter to conform).

- Why first: it is the spine every other PRD references. PRD-11 makes the
  operation/parameter model uniform and dictionary-typed; PRD-16 enforces it and,
  by refactoring the two existing transformers, *proves and de-risks* it on real
  plugins.
- Order within the stage: land PRD-11's schema/`register_operation` first, then
  PRD-16's contract + conformance test, then refactor burst/shifter onto it.
- Exit criteria: both transformers register via the uniform path; parameters are
  `.dic`-declared and validated; no bespoke transformer tables; conformance test
  gates new transformers.

## Stage 2 — Lifecycle state machine (PRD-12, LIMS P1)

- Why next: highest LIMS value, and the operation `status` folds cleanly into the
  generic transition log now that operations are uniform (Stage 1). Otherwise
  largely independent.
- Provides the auditable "where is it / how did it get here" and the QC/audit
  hooks the later layers reference.
- Could run in parallel with Stage 3 (different code area) if two people.

## Stage 3 — Study / project entity (PRD-13, LIMS P2)

- Why here: second-highest LIMS value; organizational grouping with own+public
  scoping and a dataset-browser study facet. Independent of the spine, so it may
  run in parallel with Stage 2.
- Backfills `mfdb_study` from existing `project_id` values.

## Stage 4 — Protocols (PRD-14, LIMS P3)

- Why after the spine: depends on PRD-11's `mfdb_operation_parameter_def` and
  references `operation_type`; pairs with the setup/calibration versioning and
  benefits from the lifecycle (Stage 2) for version audit.
- Cohesion note: PRD-14 extends the *same* operation/parameter code as Stage 1, so
  if you prefer cohesion over the P2-before-P3 value order, do PRD-14 immediately
  after Stage 1 and PRD-13 last of the LIMS layers. Either is fine.

## Stage 5 — Reagent inventory (PRD-15, LIMS P4)

- Lowest priority; orthogonal linking table, no schema pollution. Do when MFDB is
  shared across users or strict consumable reproducibility is required.

## Dependency graph

```
PRD-03/04/09/10 (done)
        │
        ▼
   ┌─ PRD-11 ──► PRD-16        (Stage 1, spine; ship together)
   │     │
   │     ├──────► PRD-14       (Stage 4; needs PRD-11 param schemas)
   │
   ├──► PRD-12                 (Stage 2; independent, status folds in)
   ├──► PRD-13                 (Stage 3; independent, browser facet)
   └──► PRD-15                 (Stage 5; independent, lowest priority)
```

## Parallelization

- Single track: 1 → 2 → 3 → 4 → 5.
- Two tracks after Stage 1: **A** = PRD-12 then PRD-14 (provenance/process spine);
  **B** = PRD-13 then PRD-15 (organization/inventory). They touch disjoint code.

## Recommended first action

Start **PRD-11 + PRD-16 together** — schema + `register_operation` + the
transformer contract — and refactor Burst Selection and Microtime Shifter as the
two reference conformant transformers. Everything else slots onto that spine.
