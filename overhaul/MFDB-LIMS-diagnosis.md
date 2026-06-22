# MFDB ↔ LIMS design diagnosis (vs. iSkyLIMS)

Diagnosis of how MFDB can benefit from LIMS design principles, using
`thirdparty/iskylims` (a Django LIMS for sequencing data) as the reference.

## 1. What a LIMS does (iSkyLIMS principles)

iSkyLIMS tracks samples from wet-lab through sequencing to delivery. Reading its
domain model (`core/`, `wetlab/`, `drylab/`, `clinic/models.py`) the recurring
LIMS design principles are:

1. **Lifecycle state machines with history.** Almost every entity has an explicit
   state and a state table: `StatesForSample`, `StatesForMolecule`, `PoolStates`,
   `RunStates`, `LibPrepareStates`, `ServiceState`, `ResolutionStates`,
   `ClinicSampleState`. A run moves recorded → sample-sheet-sent → processing →
   running-stats → completed; each transition is recorded (who/when). **Status is
   a tracked process, not a flag.**
2. **Protocols as typed parameter schemas.** `ProtocolType` → `Protocols` →
   `ProtocolParameters`; values recorded per item (`MoleculeParameterValue`,
   `LibParameterValue`, `ParameterPipeline`). A *protocol* names a procedure and
   defines the parameters it requires; each execution records values against it.
3. **A physical preparation chain.** `Samples` → `MoleculePreparation` →
   `LibPrepare` → `LibraryPool` → `RunProcess`. Each prep step is an entity with
   its own state, kit, operator and parameters — full bench-to-data provenance.
4. **Projects / investigations with configurable fields.** `SampleProjects` +
   `SampleProjectsFields` + `SampleProjectsFieldsValue`: samples are grouped into
   projects, and each project defines its own **user-configurable metadata
   fields**. Studies organize work and carry custom schemas.
5. **Inventory: kits, lots, expiry.** `CommercialKits`, `UserLotCommercialKits`,
   `LibraryKit`, `CollectionIndexKit`, `SequencerInLab`. Consumables and
   instruments are tracked with lots/expiry so a result ties to exactly what was
   used.
6. **Service request → resolution → delivery.** `Service`,
   `RequestedSamplesInServices`, `Resolution`, `Delivery`, `Pipelines`: the dry
   lab connects researchers to analyses through a request workflow with states.
7. **Statistics, QC and reporting.** `StatsRunSummary/Read`, `StatsLaneSummary`,
   `GraphicsStats`, annual/monthly reports, QC gating per run/sample/project.
8. **Configurability / ontologies.** `OntologyMap`, `ConfigSetting`,
   `SamplesProjectsTableOptions` make the deployment adaptable.

## 2. Where MFDB already stands (and exceeds a classic LIMS)

MFDB is a **provenance-first scientific data system**. On several axes it is
*stronger* than iSkyLIMS, and those strengths should be preserved:

- **Dictionary-dictates-schema (flrCIF/pdbx).** MFDB's `.dic` + generator +
  `validate_mapping` gate make the schema standards-aligned and self-describing.
  iSkyLIMS uses ad-hoc Django models. **Keep this as the governing principle for
  anything added below.**
- **Content-addressed object store with dedup** (`mfdb_object.content_md5`,
  refcounting) — stronger than file-path/mass-storage tracking.
- **Fine-grained provenance graph** — `mfdb_operation`, `mfdb_edge`
  (`derived_from`, `measured_sample`, `grouped_in`), parameters per operation.
  iSkyLIMS provenance is essentially linear run tracking.
- **Branching** (`mfdb_branch`), **multi-owner + ACL**
  (`mfdb_artifact_owner`, `mfdb_acl_entry`, `mfdb_group`), **time-versioned
  calibration** (`mfdb_setup_calibration`), **extensible vocabulary**
  (`mfdb_vocabulary`).

## 3. Gap analysis — LIMS principles MFDB lacks

Verified against `schema.py`: MFDB has no project/study entity (only a loose
`project_id TEXT`), no lifecycle state machine (only flat `status` /
`validation_status` fields), and no protocol, inventory, QC, or request entities.

| LIMS principle | MFDB today | Gap / opportunity |
|---|---|---|
| Lifecycle state machine + history | `mfdb_operation.status` enum; `mfdb_artifact.validation_status`; `flr_experiment.status` — flat flags, no transition history | **No tracked lifecycle** for sample / experiment / dataset (e.g. registered → measured → processed → validated → archived) with who/when transitions |
| Protocols as parameter schemas | Setups (instrument config) + per-operation parameters; no named procedure | **No protocol entity**: "how was this measured/processed" as a named, versioned, parameter-schema'd procedure |
| Preparation chain | `flr_sample_probe`, `flr_sample_condition` (static) | Labeling/prep steps not modelled as tracked, parameterized operations in the provenance graph |
| Projects / studies + custom fields | `project_id TEXT` (no table) | **No study entity** to group samples/datasets, and no per-project configurable metadata fields |
| Inventory (kits / lots / expiry) | Instruments/setups only | Consumables (fluorophore lots, buffers, filters) not inventoried → reproducibility gap |
| Request → resolution → delivery | none | No way to request an analysis / deliver results between users in a shared MFDB |
| Statistics / QC / reporting | results stored; no aggregates | No QC flags, no per-setup drift / per-sample dataset reports, no dashboards |

## 4. Recommendations (prioritized, dictionary-driven)

Adopt the LIMS principles that fit MFDB's provenance-first model; do **not** clone
iSkyLIMS's Django-shaped tables. Everything below is `.dic`-declared, generated,
and gate-covered — consistent with the rest of PRD-04/10.

### P1 — Lifecycle state machine + transition history (highest value)

The single biggest LIMS principle MFDB lacks. Add a generic, dict-declared
**state-transition log** rather than per-entity status columns:

- `mfdb_state_transition(id, entity_type, entity_id, from_state, to_state,
  reason, operator_user_id, created_at)`, plus a small `state` vocabulary per
  `entity_type` in `mfdb_vocabulary` (reuse the extensible-vocab machinery).
- Define lifecycles: **sample** (registered → measured → processed → validated →
  archived), **dataset/artifact** (registered → validated → published/archived),
  **operation** (already pending/running/succeeded/failed — fold in).
- Gives auditable "where is this / how did it get here," QC gating, and reporting
  hooks — the defining LIMS capability. Builds on the existing `mfdb_audit_log`
  but is queryable per entity.

### P2 — Study/project entity with configurable fields

Promote `project_id` from a loose string to a real grouping, mirroring
iSkyLIMS `SampleProjects` + configurable fields but dictionary-driven:

- `mfdb_study(study_id, name, description, owner, is_public, created_at, …)` and
  `mfdb_study_member(study_id, sample_id|artifact_id)`.
- Reuse the existing dynamic-field pattern (`flr_sample_key_value`,
  `mfdb_*_key_value`) for per-study custom metadata rather than a new EAV stack.
- Lets the dataset browser filter/scope by study (a natural next facet alongside
  sample, kind, owner).

### P3 — Protocol entity (named, versioned procedures)

Generalize the setup/calibration work into a **protocol** concept: a named,
versioned procedure (measurement or processing) with a declared parameter
schema; operations reference the protocol + version they ran. This formalizes
reproducibility ("re-run protocol X v3 on sample Y") and pairs naturally with the
already-implemented time-versioned calibration.

### P4 — Lightweight inventory (lots/consumables)

For reproducibility, a minimal `mfdb_reagent_lot(lot_id, kind, name, lot_number,
expiry, …)` referenced from operations/setups (e.g. which fluorophore lot, which
filter set). Lower priority for a single-lab tool; valuable if MFDB is shared.

### P5 — QC / stats / reporting layer

Aggregate views over the provenance graph: per-setup calibration drift, per-sample
dataset/operation counts, validation-state dashboards, and a QC flag on artifacts
beyond `validation_status`. Read-mostly; can be RPC handlers over existing tables.

### P6 — Request/service workflow (only if MFDB becomes multi-user/shared)

iSkyLIMS's request → resolution → delivery is valuable when researchers and
analysts are different people. For chisurf-as-a-tool it is likely out of scope;
revisit if MFDB is deployed as a shared lab service.

## 5. Summary

MFDB already implements the **data** half of a LIMS better than iSkyLIMS
(standards-driven schema, content-addressed storage, provenance graph, ownership,
versioning). What it lacks is the LIMS **process/workflow** half. In priority
order, the highest-leverage adoptions are: **(P1) lifecycle state machines with
transition history**, **(P2) a study/project entity with configurable fields**,
and **(P3) protocols as versioned parameter schemas** — each implemented the
MFDB way (dictionary-declared, generated, gate-covered), not as Django-style
tables. Inventory, QC/reporting, and request workflows follow as the deployment
grows toward a shared lab service.
