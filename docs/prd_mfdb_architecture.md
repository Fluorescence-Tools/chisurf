# Product Requirements Document: MFDB Architecture

Status date: 2026-06-12

This document supersedes the older `fdb` architecture documents. The canonical
product name is **MFDB**, the Multiparameteric Fluorescence Database.

MFDB is still pre-production. Breaking changes are allowed and preferred when
they remove ambiguity before the first production dataset exists. Do not carry
compatibility layers forward unless this PRD explicitly requires them.

## Purpose

MFDB is the provenance and archive database for multiparameter fluorescence
workflows in ChiSurf. It must preserve how scientific results were produced:
which samples, measurements, files, settings, setup snapshots, software
versions, operations, outputs, fits, parameters, project snapshots, and archives
belong together.

MFDB is not an object database and not a raw photon data store. The correct
architecture is a relational provenance core with object-like project or fit
state stored as versioned artifact payloads.

The target mental model is:

```text
Sample / Experiment / Setup
          |
       Operation
      /         \
 input artifacts output artifacts
          |
   Parameters / Audit
```

In shorter graph form:

```text
Artifact -> Operation -> Artifact
              |
       Parameters / Setup / Audit
```

## Product Goals

- Make one developer-facing architecture for MFDB.
- Remove the current split between old sample database tables, `fdb`, `mfdb`,
  legacy provenance edges, and duplicate repository classes.
- Make new workflows extend artifacts, operations, parameters, setup snapshots,
  and graph relationships instead of adding new provenance schemas.
- Keep large measurement data external by default and store references,
  checksums, validation state, and metadata in SQLite.
- Store restorable project/fitting/UI state as versioned artifacts, not as the
  database's organizing model.
- Keep the core usable from the desktop app, headless JSON-RPC/ZMQ, CLI scripts,
  and a future web service without changing the data model.

## Review Conclusions

- The primary MFDB store should be relational SQLite, not an object database.
- Python/project/fit/UI objects should be serialized as versioned artifact
  payloads at the boundary of the provenance model.
- New MFDB databases are initialized with canonical `mfdb_*` provenance
  tables plus the accepted `flr_*` and PDBx/PDB-IHM fluorescence domain
  tables. Experimental `fdb_*` tables are migration/import artifacts only.
- The current transition-friendly shape is too broad for production. The next
  implementation should hard-cut to one package, one schema authority, one
  repository class, one service namespace, and one graph source of truth.

## Non-Goals

- Do not use an object database as the primary store.
- Do not store large TTTR/photon streams in SQLite by default.
- Do not remove `flr_*`, PDBx, or PDB-IHM support. Keep pre-production
  `fdb.v1.*` APIs and old `fdb_*` migration tables out of the production
  contract.
- Do not rewrite Burst Selection, ndxplorer, fitting, or archive algorithms.
  MFDB records their inputs, settings, outputs, and provenance.
- Do not normalize every algorithm setting. Opaque settings JSON is acceptable
  until a field must be searched, compared, restored independently, or linked.

## Architecture Decisions

### Naming

- Canonical package: `chisurf.core.mfdb`.
- Canonical repository class: `MFDatabase`.
- Canonical service namespace: `mfdb.v1.*`.
- Canonical SQLite tables use the `mfdb_` prefix for provenance core tables and
  `flr_*`/PDBx/PDB-IHM names for accepted fluorescence domain tables.
- Remove `fdb.v1.*` service aliases before production.
- Remove `FluorophoreDatabase` and `FluorescenceDatabase` from the MFDB public
  API. If unrelated code still needs those names, keep shims outside the MFDB
  core and mark them for deletion.
- Remove or replace `chisurf.core.fio.mmcif.db.repository` as an MFDB authority.
  mmCIF/FLR import-export helpers may remain there only if they call
  `chisurf.core.mfdb.MFDatabase` and do not define a second repository class.

### Storage Model

Use SQLite as the primary metadata and provenance store.

Do not introduce an object database as the primary MFDB store. Object databases
optimize persistence of in-memory application graphs, but MFDB's durable
contract is cross-version provenance, queryable scientific metadata, archive
exchange, and service portability. Those requirements fit a relational
provenance core better than object persistence.

Use external files for large/raw data:

- local files and folders
- URLs
- managed archive members
- embedded JSON or blobs only for small payloads

Use JSON payloads for project snapshots, settings, model state, fit structures,
and UI restore data. These JSON payloads are stored as artifacts and referenced
by checksum. They are not the schema authority.

Every object snapshot artifact must use a small envelope so future restore code
can reject incompatible payloads before touching application state:

```json
{
  "snapshot_schema": "chisurf.project",
  "snapshot_schema_version": 1,
  "object_type": "project_snapshot",
  "producer": {"package": "chisurf", "module": "...", "version": "..."},
  "payload": {}
}
```

The `payload` may contain rich nested project state. Queryable scientific
identity, provenance, parameters, checksums, and operation context must remain
in canonical MFDB tables.

### Identity and Versioning

- Use stable text IDs for all public entities. Generated IDs should be UUID or
  UUID-prefixed strings, not row-number-dependent identifiers.
- Artifact content is immutable once a checksum-backed artifact has downstream
  links. A changed file, snapshot, or result gets a new artifact ID.
- Operation settings are immutable once an operation reaches `succeeded`,
  `failed`, or `cancelled`. Status and error fields may change only through a
  status-transition method.
- Setup records are immutable snapshots. Changing a setup creates a new
  `setup_version`.
- Project snapshots are append-only. A replacement snapshot links to the earlier
  snapshot with a `supersedes` edge.
- Deletes should be rare. Prefer `superseded` or `archived` metadata for
  scientific records unless a test or explicit cleanup operation needs hard
  deletion.

### Canonical Entity Model

MFDB has two layered entity models:

Core provenance entities:

- `Sample`: what was measured, described with stable ID and metadata.
- `Experiment`: user/project context grouping measurements and analyses.
- `Setup`: immutable or versioned instrument/configuration snapshot.
- `Artifact`: a data object, file reference, folder reference, object snapshot,
  exported archive, figure, table, fit result, or manifest.
- `Operation`: a measurement, import, processing, analysis, fitting, archive, or
  restore action.
- `OperationArtifact`: the canonical input/output link between operations and
  artifacts.
- `Parameter`: queryable values, constraints, fit parameters, uncertainty, and
  dependency metadata linked to an operation.
- `Edge`: graph relationships that are not operation inputs or outputs.
- `AuditLog`: database mutation history.

Scientific domain entities:

- `flr_*` sample, probe, instrument, setup, experiment, and analysis tables.
- PDBx/PDB-IHM mmCIF-compatible tables used for structure, fluorophore, and
  experiment import/export.

### Canonical Tables

New MFDB databases create canonical provenance core tables, required indices,
and accepted fluorescence domain tables:

```text
mfdb_schema_version
mfdb_setup
mfdb_artifact
mfdb_operation
mfdb_operation_artifact
mfdb_parameter
mfdb_edge
mfdb_audit_log
mfdb_vocabulary
flr_sample
flr_sample_condition
flr_sample_users
flr_sample_devices
flr_experiment_type
flr_experiment
...
PDBx/PDB-IHM mmCIF-compatible domain tables
```

The implementation may add specialized tables later only when the data is
stable, queried frequently, and clearly belongs to one canonical entity. Examples
that may become justified later are covariance blocks, artifact storage
locations, archive manifests, and setup components.

### Table Responsibilities

`flr_sample`

- Stores sample identity and broad biological/chemical context for fluorescence
  experiments.
- Required fields are inherited from the mmCIF FLR category definitions,
  including `sample_id`, descriptive metadata, probe/device/user links, and
  project context.

`flr_experiment`

- Groups fluorescence measurements, analyses, samples, and archived projects.
- Required fields are inherited from the mmCIF FLR category definitions,
  including `experiment_id`, sample linkage, status, and experiment metadata.

`mfdb_setup`

- Stores immutable or versioned setup snapshots.
- Required fields: `setup_id`, `version`, `name`, `instrument_id`,
  `configuration_json`, `created_at`, `updated_at`.
- `(setup_id, version)` identifies the exact configuration used by an operation.

`mfdb_artifact`

- Stores artifact identity, semantic kind, data format, storage reference,
  checksum, validation, and metadata.
- Required fields: `artifact_id`, `artifact_kind`, `data_format`,
  `storage_mode`, `checksum`, `checksum_algorithm`, `validation_status`,
  `metadata_json`, `created_at`, `updated_at`.
- Optional storage fields: `file_path`, `folder_path`, `url`,
  `archive_member_path`, `data_json`, `data_blob`.

`mfdb_operation`

- Stores reproducible actions.
- Required fields: `operation_id`, `operation_type`, `status`,
  `settings_json`, `settings_hash`, `software_package`, `software_module`,
  `software_version`, `runtime_environment_json`, `created_at`, `updated_at`.
- Optional context fields: `experiment_id`, `setup_id`, `setup_version`,
  `operator_user_id`, `started_at`, `ended_at`, `error_message`.

`mfdb_operation_artifact`

- Is the only source of truth for operation input/output relationships.
- Required fields: `operation_id`, `artifact_id`, `direction`, `role`,
  `ordinal`, `checksum_snapshot`, `metadata_json`.
- Primary key: `(operation_id, artifact_id, direction, role)`.

`mfdb_parameter`

- Stores queryable parameters and dependencies for operations.
- Required fields: `parameter_id`, `parameter_uuid`, `operation_id`, `name`,
  `parameter_type`, `value`, `units`, `bounds_on`, `metadata_json`.
- Optional fields include standard error, confidence interval, initial value,
  lower/upper bounds, expression, prior JSON, mapping JSON, and covariance
  reference.

`mfdb_edge`

- Stores graph relationships that are not operation inputs or outputs.
- Allowed examples: `included_in`, `contains`, `derived_from`, `supersedes`,
  `uses_external_reference`, `parameter_depends_on`, `project_contains`,
  `grouped_in`.
- Do not store `input_to` or `produced` in `mfdb_edge`; those belong only in
  `mfdb_operation_artifact`.
- A read-only graph export/view may synthesize `input_to` and `produced` edges
  from `mfdb_operation_artifact`, but the synthesized edges must not be written
  back into `mfdb_edge`.

`mfdb_audit_log`

- Stores mutation records: action, target type, target ID, operator, timestamp,
  and details JSON.
- Audit explains database changes. Provenance explains scientific/data lineage.

`mfdb_vocabulary`

- Stores extensible vocabulary values that are allowed for a controlled field.
- Required fields: `field_name`, `value`, `display_name`, `description`,
  `is_builtin`, `is_active`.

## Legacy Cutover Policy

Because MFDB is not in production, the implementation should not maintain a
general migration waterfall from all historical experimental schemas. Instead:

- New database creation uses canonical `mfdb_*` provenance tables plus accepted
  `flr_*` and PDBx/PDB-IHM domain tables.
- Existing pre-production `fdb_*` sample databases may be discarded,
  regenerated, or imported through a one-time script.
- If old `fdb_*` data must be kept for tests, write an explicit importer that
  reads the legacy tables and writes canonical MFDB records through
  `MFDatabase`.
- Do not dual-write legacy and canonical tables.
- Do not keep a legacy provenance table as a second graph source.
- Do not keep both old and new repository classes in importable production code.

### Vocabulary Policy

Use stable small vocabularies with SQLite `CHECK` constraints:

- `direction`: `input`, `output`
- `status`: `pending`, `running`, `succeeded`, `failed`, `cancelled`
- `validation_status`: `unvalidated`, `valid`, `invalid`, `warning`
- `storage_mode`: `local_file`, `local_directory`, `url`, `managed_archive`,
  `embedded_json`, `embedded_blob`

Use `mfdb_vocabulary` plus repository validation for extensible vocabularies:

- `artifact_kind`
- `data_format`
- `operation_type`
- `parameter_type`
- `relationship_type`

Built-in artifact kinds:

- `raw_measurement`
- `processed_data`
- `analysis_result`
- `fit_result`
- `parameter_table`
- `selection_mask`
- `project_snapshot`
- `archive_manifest`
- `archive_file`
- `visualization`
- `external_reference`

Built-in data formats:

- `ptu`, `spc`, `bh`, `tttr`, `photon_hdf5`, `bur`, `hdf5`, `zip`, `json`,
  `csv`, `tsv`, `png`, `svg`, `sqlite`, `directory`, `unknown`

Built-in operation types:

- `measurement_import`
- `validation`
- `burst_selection`
- `filtering`
- `fcs_correlation`
- `microtime_histogram`
- `tcspc_fitting`
- `model_fitting`
- `ndxplorer_selection`
- `ndxplorer_clustering`
- `project_snapshot`
- `project_restore`
- `archive_export`

## Public API Requirements

### Python API

The repository class must expose focused methods around the canonical model:

- `MFDatabase(path)`
- `MFDatabase.create_empty(path)`
- `MFDatabase.open_existing(path)`
- `register_sample(...)`
- `register_experiment(...)`
- `save_setup_snapshot(...)`
- `register_artifact(...)`
- `record_operation(...)`
- `transition_operation_status(...)`
- `link_operation_artifact(...)`
- `record_operation_with_artifacts(...)`
- `record_parameter(...)`
- `add_edge(...)`
- `get_artifact(...)`, `list_artifacts(...)`
- `get_operation(...)`, `list_operations(...)`
- `graph_upstream(node_type, node_id, max_depth=100)`
- `graph_downstream(node_type, node_id, max_depth=100)`
- `export_graph(...)`
- `list_audit_logs(...)`

`record_operation_with_artifacts(...)` is the preferred high-level write API for
workflow adapters. It must create/update the operation, register declared output
artifacts, link inputs and outputs, record parameters, write audit logs, and
commit or roll back as one transaction.

Adapters should not call low-level SQL or assemble partial provenance manually.
Direct `record_operation(...)` is allowed only for operations without artifacts
or as an internal step inside `record_operation_with_artifacts(...)`.

The repository should expose query results as dictionaries or small typed data
objects. It must not return raw `sqlite3.Row` objects from public MFDB methods.

### JSON-RPC API

Register only `mfdb.v1.*` methods:

```text
mfdb.v1.samples.register
mfdb.v1.samples.get
mfdb.v1.samples.list
mfdb.v1.experiments.register
mfdb.v1.experiments.get
mfdb.v1.experiments.list
mfdb.v1.setups.save
mfdb.v1.setups.get
mfdb.v1.setups.list
mfdb.v1.artifacts.register
mfdb.v1.artifacts.get
mfdb.v1.artifacts.list
mfdb.v1.operations.record
mfdb.v1.operations.get
mfdb.v1.operations.list
mfdb.v1.operations.link_artifact
mfdb.v1.operations.record_with_artifacts
mfdb.v1.parameters.record
mfdb.v1.parameters.list
mfdb.v1.graph.upstream
mfdb.v1.graph.downstream
mfdb.v1.graph.export
mfdb.v1.archives.export
mfdb.v1.projects.snapshot
mfdb.v1.projects.restore
mfdb.v1.audit.list
```

Legacy plugin methods may remain as adapters only if existing GUI code still
needs them. Adapters must call `MFDatabase` methods and must not assemble
provenance rows manually.

Legacy adapter methods must be visibly separated from canonical service
registration, for example in `compatibility_services.py`. They must not be mixed
into the canonical `mfdb.v1.*` registration table.

## Workflow Contracts

### Burst Selection

The Burst Selection adapter must:

- Register the raw TTTR/PTU/SPC/BH file as an artifact with
  `artifact_kind = raw_measurement`.
- Record one `burst_selection` operation with settings, settings hash, software
  identity, status, timing, and runtime environment.
- Link the raw artifact as operation input.
- Register `.bur`, HDF5, ZIP, JSON summary, and archive manifest outputs as
  artifacts.
- Link all outputs through `mfdb_operation_artifact`.
- Store any restorable Burst Selection state as a `project_snapshot` or
  `analysis_result` artifact when needed.

### ndxplorer

The ndxplorer adapter must:

- Consume registered artifacts, not file paths passed around outside MFDB.
- Record selections, clustering, masks, summaries, and exported plots/tables as
  outputs of explicit operations.
- Avoid GUI imports in service code. Headless reader/analysis modules are the
  only allowed service dependencies.

### Fitting and Project Archive

Project and fitting state must be represented in two layers:

- Queryable fit parameters and dependencies go into `mfdb_parameter`.
- Full restorable project, model, widget, and UI state goes into a JSON artifact
  with `artifact_kind = project_snapshot`.

The project snapshot artifact must link downstream from the operations and
artifacts it depends on. Restoring a project must keep stored UUIDs so a later
save can supersede or version the previous snapshot instead of creating an
unrelated record.

## Required Invariants

- One canonical repository implementation exists.
- One canonical schema module exists.
- New MFDB databases contain `mfdb_*` provenance core tables plus accepted
  `flr_*` and PDBx/PDB-IHM domain tables.
- All public mutations run inside one transaction.
- Audit records are part of the same transaction as the mutation.
- Operation input/output relationships are stored only in
  `mfdb_operation_artifact`.
- `mfdb_edge` never stores `input_to` or `produced`.
- Direct SQLite writes cannot insert invalid stable vocabulary values.
- Repository methods reject inactive or unknown extensible vocabulary values.
- Re-saving an operation must not delete existing inputs, outputs, parameters, or
  graph history unless an explicit delete or supersede API is called.
- Hard-deleting an operation must remove operation links, parameters,
  operation-scoped non-operation edges, and audit-visible state consistently.
- Superseding an operation or artifact must create a new record and connect the
  old and new records with a `supersedes` edge.
- Checksums are required for file-backed artifacts unless an adapter explicitly
  marks the artifact `unvalidated` and records why.
- All JSON fields must be valid JSON text. Repository methods must serialize and
  deserialize them centrally.
- All timestamps must be timezone-aware UTC ISO-8601 strings.

## Implementation Plan

### Phase 1 - Canonical Core Reset

- Rename the architecture and docs from `fdb` to `mfdb`.
- Keep `chisurf.core.mfdb`; make it the only database core.
- Rename the repository class to `MFDatabase`.
- Create canonical schema creation for `mfdb_*` provenance tables plus accepted
  `flr_*` and PDBx/PDB-IHM domain tables.
- Remove duplicated old repository/schema authority from
  `chisurf.core.fio.mmcif.db`; no second `FluorescenceDatabase` or
  `FluorophoreDatabase` class may remain as an MFDB implementation.
- Replace `fdb.v1.*` registrations with `mfdb.v1.*` only.
- Split canonical services from compatibility wrappers.

### Phase 2 - Repository and Invariants

- Implement a unit-of-work helper used by every mutation.
- Implement vocabulary bootstrap and validation.
- Add SQLite constraints for stable vocabularies.
- Implement `record_operation_with_artifacts(...)` as the default workflow write
  path.
- Implement graph traversal from `mfdb_operation_artifact` plus non-operation
  `mfdb_edge`.
- Add delete/supersede behavior that cannot leave orphan links, parameters, or
  edges.
- Centralize JSON serialization, timestamp generation, settings hashing, and
  checksum calculation.

### Phase 3 - Adapter Migration

- Move Burst Selection persistence to the canonical operation/artifact API.
- Move ndxplorer persistence to the canonical operation/artifact API.
- Move project archive/restore to project snapshot artifacts plus queryable
  parameters.
- Keep legacy GUI/service entrypoints as wrappers only where needed by current
  UI code.
- Replace file-path handoffs between adapters with artifact IDs wherever the
  producer and consumer are both MFDB-aware.

### Phase 4 - Cleanup

- Delete or quarantine old `fdb_*` MFDB tables from new database creation while
  keeping `flr_*` and PDBx/PDB-IHM domain support.
- Remove stale `fdb` docs or mark them superseded by this PRD.
- Update examples and notebooks to import `MFDatabase` from `chisurf.core.mfdb`.
- Ensure the plugin manifest lists all registered `mfdb.v1.*` methods and no
  stale `fdb.v1.*` methods.
- Add a repo-wide check that no production code imports old MFDB authority paths.

## Test Plan

### Schema Tests

- Creating a new database creates canonical `mfdb_*` provenance tables plus
  accepted `flr_*` and PDBx/PDB-IHM domain tables.
- Schema version is stored in `mfdb_schema_version`.
- No experimental `mfdb_sample`, `mfdb_experiment`, old `fdb_*`, or sample-database
  provenance tables are created by the canonical schema.
- Stable vocabulary `CHECK` constraints reject invalid `direction`, `status`,
  `validation_status`, and `storage_mode`.
- Extensible vocabulary validation rejects unknown or inactive artifact kinds,
  data formats, operation types, parameter types, and relationship types.
- Direct SQL cannot insert `input_to` or `produced` into `mfdb_edge`.
- All foreign keys are enabled and enforced for repository-created connections.

### Repository Tests

- `record_operation_with_artifacts(...)` creates operation, artifacts, links,
  parameters, and audit log in one transaction.
- Simulated audit failure rolls back the whole mutation.
- Simulated link failure rolls back operation and artifact writes.
- Re-saving an operation updates operation metadata without deleting links or
  parameters.
- Delete/supersede leaves no orphan operation links, parameters, or edges.
- Public methods return dicts or typed objects, not raw `sqlite3.Row` values.
- JSON fields roundtrip through one serializer and reject invalid JSON on write.
- Terminal operation settings cannot be mutated without a supersede/new
  operation path.

### Graph Tests

- Upstream traversal from a `.bur` artifact reaches the producing Burst Selection
  operation and raw TTTR artifact.
- Downstream traversal from a raw artifact reaches all derived outputs.
- Non-operation edges such as `included_in` and `supersedes` are included.
- `input_to` and `produced` never appear as duplicate rows from `mfdb_edge`.
- Cycle handling terminates and reports each semantic edge once.
- A graph export can synthesize `input_to` and `produced` from operation links
  without writing those relationship types into `mfdb_edge`.

### Adapter Tests

- Burst Selection roundtrip records raw input, operation, outputs, checksums, and
  archive manifest through the canonical API.
- ndxplorer service tests run headlessly and record selections/clustering as
  canonical operations and artifacts.
- Project snapshot/restore stores restorable JSON as an artifact and queryable
  parameters in `mfdb_parameter`.
- Project snapshot payloads include `snapshot_schema`,
  `snapshot_schema_version`, `object_type`, `producer`, and `payload`.
- A changed project snapshot creates a new artifact and a `supersedes` edge.

### Service Tests

- Registered service methods exactly match the plugin manifest.
- Only `mfdb.v1.*` methods are registered for the canonical MFDB API.
- Legacy wrappers, if retained, call canonical methods and do not write tables
  directly.
- Canonical services and compatibility wrappers are registered from separate
  modules.
- Repo search tests fail on production imports of old MFDB authority paths.

### Architecture Drift Tests

- There is exactly one importable canonical repository class:
  `chisurf.core.mfdb.MFDatabase`.
- `chisurf.core.fio.mmcif.db.repository` does not define an MFDB repository
  class.
- No production service registration includes `fdb.v1.*`.
- No canonical MFDB method dual-writes legacy and canonical tables.

## Acceptance Criteria

- A coding agent can explain MFDB using only the canonical entities in this PRD.
- A new SQLite database contains no legacy FDB or experimental MFDB sample tables;
  accepted `flr_*` and PDBx/PDB-IHM domain tables are first-class MFDB storage.
- There is exactly one canonical repository class for MFDB.
- There is exactly one canonical schema module for MFDB.
- Burst Selection, ndxplorer, fitting, and project archive all persist through
  the same artifact-operation-artifact model.
- Object-like project state is stored only as versioned artifacts and never
  becomes the primary database model.
- `test/fio/test_mfdb_*.py` or equivalent tests cover schema, transactions,
  graph traversal, adapters, and service manifest consistency.
- The implementation can be reviewed without reconciling multiple database
  names, multiple repository classes, or multiple graph storage models.
