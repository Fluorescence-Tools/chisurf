---
type: Reference
title: MFDB ↔ LIMS design diagnosis
description: Gap analysis of MFDB against laboratory information-management-system design principles, mapping each gap to PRD-12 through PRD-15.
tags: [reference, mfdb, lims]
timestamp: '2026-07-06T00:00:00Z'
---

# Purpose

This concept preserves the durable gap analysis behind the MFDB LIMS-workflow
PRDs. It asks: what does a mature **laboratory information-management system
(LIMS)** provide that MFDB — ChiSurf's SQLite-backed metadata and provenance
store (`chisurf/core/mfdb/`) — does not yet, and how does each gap map onto a
concrete PRD? The comparison uses an established open-source LIMS built for
sequencing/wet-lab workflows as the reference point for LIMS design principles;
that product is described here by role rather than by name.

The takeaway: MFDB already implements the **data** half of a LIMS more strongly
than a classic LIMS does, but lacks most of the **process/workflow** half. The
four highest-leverage adoptions are captured as PRDs:

- Lifecycle state machines with transition history → [PRD-12](/prds/prd-12.md)
- Study/project entity with configurable fields → [PRD-13](/prds/prd-13.md)
- Protocols as named, versioned parameter schemas → [PRD-14](/prds/prd-14.md)
- Lightweight inventory (lots/consumables) → [PRD-15](/prds/prd-15.md)

# What a LIMS does (design principles)

A LIMS tracks samples from wet-lab through instrument runs to delivery. The
recurring design principles observed in a mature reference LIMS are:

1. **Lifecycle state machines with history.** Almost every entity has an
   explicit state plus a state-transition table (sample states, molecule states,
   pool states, run states, library-prep states, service/resolution states). A
   run moves recorded → sample-sheet-sent → processing → running-stats →
   completed, and each transition records who and when. **Status is a tracked
   process, not a flag.**
2. **Protocols as typed parameter schemas.** A protocol type owns protocols,
   which own protocol parameters; values are recorded per item against the
   protocol. A protocol names a procedure and declares the parameters it
   requires; each execution records values against it.
3. **A physical preparation chain.** Samples → molecule preparation → library
   prep → pool → run process. Each prep step is an entity with its own state,
   kit, operator and parameters — full bench-to-data provenance.
4. **Projects / investigations with configurable fields.** Samples are grouped
   into projects, and each project defines its own **user-configurable metadata
   fields** with per-field values. Studies organize work and carry custom
   schemas.
5. **Inventory: kits, lots, expiry.** Commercial kits, user lots, library kits,
   index kits, in-lab instruments. Consumables and instruments are tracked with
   lots/expiry so a result ties to exactly what was used.
6. **Service request → resolution → delivery.** A request workflow with states
   connects researchers to analyses (service → requested samples → resolution →
   delivery → pipelines).
7. **Statistics, QC and reporting.** Run/lane/read summaries, graphics,
   periodic reports, QC gating per run/sample/project.
8. **Configurability / ontologies.** Ontology maps, config settings and table
   options make the deployment adaptable.

# Where MFDB already stands (and exceeds a classic LIMS)

MFDB is a **provenance-first scientific data system**. On several axes it is
*stronger* than the reference LIMS, and those strengths should be preserved:

- **Dictionary-dictates-schema (flrCIF/pdbx).** MFDB's `.dic` dictionaries plus
  the DDL generator and the `validate_mapping` gate make the schema
  standards-aligned and self-describing. The reference LIMS uses ad-hoc,
  framework-shaped models. **Keep dictionary-driven schema as the governing
  principle for anything added below.**
- **Content-addressed object store with dedup** (`mfdb_object.content_md5`,
  refcounting) — stronger than file-path/mass-storage tracking.
- **Fine-grained provenance graph** — `mfdb_operation`, `mfdb_edge`
  (`derived_from`, `measured_sample`, `grouped_in`), parameters per operation.
  A classic LIMS's provenance is essentially linear run tracking.
- **Branching** (`mfdb_branch`), **multi-owner + ACL** (`mfdb_artifact_owner`,
  `mfdb_acl_entry`, `mfdb_group`), **time-versioned calibration**
  (`mfdb_setup_calibration`), **extensible vocabulary** (`mfdb_vocabulary`).

# Gap analysis — LIMS principles MFDB lacks

Verified against the MFDB schema: MFDB has no project/study entity (only a loose
`project_id TEXT`), no lifecycle state machine (only flat `status` /
`validation_status` fields), and no protocol, inventory, QC, or request
entities.

| LIMS principle | MFDB today | Gap / opportunity | PRD |
|---|---|---|---|
| Lifecycle state machine + history | `mfdb_operation.status` enum; `mfdb_artifact.validation_status`; `flr_experiment.status` — flat flags, no transition history | **No tracked lifecycle** for sample / experiment / dataset (registered → measured → processed → validated → archived) with who/when transitions | [PRD-12](/prds/prd-12.md) |
| Projects / studies + custom fields | `project_id TEXT` (no table) | **No study entity** to group samples/datasets, and no per-project configurable metadata fields | [PRD-13](/prds/prd-13.md) |
| Protocols as parameter schemas | Setups (instrument config) + per-operation parameters; no named procedure | **No protocol entity**: "how was this measured/processed" as a named, versioned, parameter-schema'd procedure | [PRD-14](/prds/prd-14.md) |
| Inventory (kits / lots / expiry) | Instruments/setups only | Consumables (fluorophore lots, buffers, filters) not inventoried → reproducibility gap | [PRD-15](/prds/prd-15.md) |
| Preparation chain | `flr_sample_probe`, `flr_sample_condition` (static) | Labeling/prep steps not modelled as tracked, parameterized operations in the provenance graph | (folds into PRD-12/PRD-14) |
| Request → resolution → delivery | none | No way to request an analysis / deliver results between users in a shared MFDB; in scope only if MFDB becomes a shared lab service | — |
| Statistics / QC / reporting | results stored; no aggregates | No QC flags, no per-setup drift / per-sample dataset reports, no dashboards | — |

# How each gap maps to a PRD

Each adoption is implemented **the MFDB way** — dictionary-declared, generated,
and gate-covered — not as framework-shaped tables cloned from the reference
LIMS.

## Lifecycle state machine + transition history → [PRD-12](/prds/prd-12.md)

The single biggest LIMS principle MFDB lacks, and the highest value. Rather than
per-entity status columns, add a generic, dictionary-declared **state-transition
log**: an `mfdb_state_transition(id, entity_type, entity_id, from_state,
to_state, reason, operator_user_id, created_at)` table, plus a small `state`
vocabulary per `entity_type` reusing the extensible-vocab machinery
(`mfdb_vocabulary`). Lifecycles: **sample** (registered → measured → processed →
validated → archived), **dataset/artifact** (registered → validated →
published/archived), **operation** (fold in the existing
pending/running/succeeded/failed). This yields auditable "where is this / how did
it get here," QC gating, and reporting hooks — the defining LIMS capability —
building on `mfdb_audit_log` but queryable per entity.

## Study/project entity with configurable fields → [PRD-13](/prds/prd-13.md)

Promote `project_id` from a loose string to a real grouping. Add
`mfdb_study(study_id, name, description, owner, is_public, created_at, …)` and
`mfdb_study_member(study_id, sample_id|artifact_id)`. Reuse the existing
dynamic-field pattern (`flr_sample_key_value`, `mfdb_*_key_value`) for per-study
custom metadata rather than introducing a new EAV stack. This lets the dataset
browser filter/scope by study as a natural facet alongside sample, kind, and
owner.

## Protocol entity (named, versioned procedures) → [PRD-14](/prds/prd-14.md)

Generalize the setup/calibration work into a **protocol** concept: a named,
versioned procedure (measurement or processing) with a declared parameter
schema; operations reference the protocol + version they ran. This formalizes
reproducibility ("re-run protocol X v3 on sample Y") and pairs naturally with the
already-implemented time-versioned calibration.

## Lightweight inventory (lots/consumables) → [PRD-15](/prds/prd-15.md)

For reproducibility, a minimal `mfdb_reagent_lot(lot_id, kind, name, lot_number,
expiry, …)` referenced from operations/setups (e.g. which fluorophore lot, which
filter set). Lower priority for a single-lab tool; valuable once MFDB is shared.

## Out of scope for now — QC/reporting and request workflows

A **QC/stats/reporting** layer (per-setup calibration drift, per-sample
dataset/operation counts, validation-state dashboards, QC flags beyond
`validation_status`) is read-mostly and can be added later as RPC handlers over
existing tables. A **request → resolution → delivery** workflow is valuable only
when researchers and analysts are different people; for ChiSurf-as-a-tool it is
likely out of scope and should be revisited if MFDB is deployed as a shared lab
service. Neither has a dedicated PRD in the PRD-12…PRD-15 wave.

# Summary

MFDB already implements the **data** half of a LIMS better than the reference
LIMS (standards-driven schema, content-addressed storage, provenance graph,
ownership, versioning). What it lacks is the LIMS **process/workflow** half. In
priority order the highest-leverage adoptions — each dictionary-declared,
generated, and gate-covered — are: lifecycle state machines with transition
history ([PRD-12](/prds/prd-12.md)), a study/project entity with configurable
fields ([PRD-13](/prds/prd-13.md)), and protocols as versioned parameter schemas
([PRD-14](/prds/prd-14.md)), with lightweight inventory
([PRD-15](/prds/prd-15.md)) following. QC/reporting and request workflows follow
as the deployment grows toward a shared lab service.
