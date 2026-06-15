# Product Requirements Document: fdb Architecture Migration

Status date: 2026-06-12

## Purpose

This PRD defines the migration from the current phase-grown `fdb`
implementation to a clearer, future-proof architecture.

The target architecture is:

```text
Artifact -> Operation -> Artifact
              |
       Parameters / Setup / Software / Audit
```

`fdb` should be understood as a provenance database for fluorescence
measurement and analysis workflows. It is not primarily a file database and it
should not become a set of workflow-specific schemas that each define their own
provenance rules.

The migration must make the data model easy for developers to understand,
extend, test, and expose through desktop, JSON-RPC/ZMQ, and future web/server
interfaces.

## Problem Statement

The current implementation proves the core idea but is becoming difficult to
understand because several concepts were added phase by phase:

- raw data records
- processing runs
- processed data products
- provenance edges
- setup definitions
- analysis runs
- analysis parameters
- project archives
- ndxplorer integrations
- archive/export services
- audit logging

These pieces mostly work as vertical slices, but the architecture is not yet
centered on one small set of domain concepts. Developers currently need to
understand several overlapping models to answer simple questions:

- What is a database node?
- What is an input?
- What is an output?
- Is an analysis a processing run, an analysis run, or both?
- Is a project archive an analysis result or a project entity?
- Which API namespace is canonical?
- Which schema table should a new workflow extend?

This migration addresses those questions by defining one canonical `fdb` core.

## Goals

- Establish one developer-facing architecture for `fdb`.
- Reduce scattered workflow-specific schema concepts.
- Preserve existing user data through non-destructive migrations.
- Keep existing sample database behavior working during the transition.
- Keep existing service methods available as compatibility wrappers.
- Make new workflows extend the same artifact/operation/provenance model.
- Make provenance graph queries cycle-safe, typed, and testable.
- Make write operations transactional, idempotent, and audit-safe.
- Prepare the architecture for future server/web use without rewriting the data
  model.

## Non-Goals

- Do not rewrite Burst Selection, ndxplorer, fitting, or archive algorithms.
- Do not force raw TTTR/photon files into SQLite.
- Do not remove legacy `sample_database.*` APIs during this migration.
- Do not remove legacy FDB tables until a later cleanup phase has explicit
  migration evidence and rollback coverage.
- Do not normalize every algorithm-specific setting into relational columns.
  Structured JSON is acceptable until repeated query needs justify a table.

## Product Principles

- Provenance is the center of `fdb`.
- Large measurement data stays external by default.
- Database records describe identity, context, storage references, checksums,
  settings, software, and relationships.
- New workflows add artifact types, operation types, parameters, and adapters.
  They should not add a new provenance model.
- The repository layer owns invariants. Service handlers and GUI code should not
  assemble graph consistency by hand.
- Public service names should be versioned and discoverable.
- Documentation should separate architecture, generated schema reference, and
  generated API reference.

## Canonical Domain Model

### Artifact

An artifact is any stored, referenced, or embedded object that can be consumed or
produced by an operation.

Examples:

- raw TTTR/PTU/SPC/BH file reference
- photon-HDF5 file
- burst table
- HDF5 burst export
- FCS correlation curve
- TCSPC decay
- IRF curve
- PDA histogram
- model curve
- residual curve
- fit result
- parameter table
- ndxplorer selection mask
- plot/table export
- project archive
- archive manifest
- external file reference

Artifacts have stable identity. Updating an artifact record must not delete
downstream provenance.

### Operation

An operation is an action that consumes artifacts, context, and parameters, then
produces artifacts.

Examples:

- import
- validation
- burst selection
- filtering
- FCS correlation
- microtime histogram generation
- TCSPC fitting
- model fitting
- ndxplorer selection
- ndxplorer clustering
- project archive
- archive export

Operations are the primary unit of reproducibility. They store settings,
software identity, runtime status, operator, timestamps, errors, and references
to setup/context.

### Link

A link connects artifacts and operations.

The core operation links are:

- artifact `input_to` operation
- operation `produced` artifact

Additional graph edges may be used for relationships that are not simple
operation inputs or outputs:

- artifact `included_in` archive manifest
- operation `uses_setup` setup definition
- parameter `parameter_of` operation
- project archive `contains` fit result
- artifact `derived_from` artifact when no explicit operation exists

The implementation may store input/output links in one table with a direction
column or in separate input/output tables. The developer-facing model remains
the same.

### Parameter

A parameter describes a value, constraint, dependency, or model setting that is
semantically important for an operation.

Parameters cover:

- free, fixed, linked, shared, local, and global fit parameters
- bounds and constraints
- expressions and transformed parameters
- uncertainty estimates
- covariance/correlation information
- dataset-to-parameter mappings
- optimizer settings when queryable as parameters

Opaque operation settings may still be stored as JSON. Parameters are for
values that need identity, dependency tracking, search, comparison, or restore.

### Setup

A setup describes the measurement instrument and configuration used to produce
or interpret artifacts.

Setup definitions should be versioned or immutable snapshots. An operation
should reference the exact setup version it used.

Examples:

- instrument
- optical path
- lasers
- detectors
- filters
- timing calibration
- IRF definition
- PIE/ALEX/MFD windows
- burst defaults
- FCS calibration

### Audit

Audit records describe database changes.

Audit is not the same as provenance. Provenance explains scientific/data
lineage. Audit explains who or what changed the database and when.

## Target Storage Model

The target schema should converge on a small set of table families.

### Required Core Tables

```text
fdb_artifact
fdb_operation
fdb_operation_artifact
fdb_edge
fdb_parameter
fdb_setup
fdb_audit_log
```

`fdb_operation_artifact` is the canonical table for operation inputs and
outputs. It can store direction, role, ordinal, checksum snapshot, and optional
metadata.

`fdb_edge` is for general graph relationships that are not fully represented by
operation input/output rows, or for a compatibility graph view over the same
facts. It must not become a second, conflicting provenance model.

### Optional Specialized Tables

Specialized tables are allowed only when they add stable structure that many
features need.

Acceptable examples:

- setup components if the setup JSON becomes too large or frequently queried
- parameter covariance blocks
- archive manifests
- artifact storage locations

Specialized tables should reference canonical artifact, operation, setup, or
parameter IDs. They should not introduce alternative identities for the same
concept.

## Target API Shape

All new public methods should live under a versioned namespace:

```text
fdb.v1.artifacts.register
fdb.v1.artifacts.get
fdb.v1.artifacts.list
fdb.v1.operations.record
fdb.v1.operations.get
fdb.v1.operations.list
fdb.v1.graph.upstream
fdb.v1.graph.downstream
fdb.v1.graph.export
fdb.v1.parameters.record
fdb.v1.setups.save
fdb.v1.setups.get
fdb.v1.archives.export
fdb.v1.projects.archive
fdb.v1.projects.restore
fdb.v1.audit.list
```

Existing service methods should remain as compatibility wrappers:

- `raw_data.*`
- `processing.*`
- `processed_data.*`
- `provenance.*`
- `archive.*`
- `project.*`
- `database.*`
- `ndxplorer.*`
- `sample_database.setups.*`

The wrappers should call the canonical `fdb.v1.*` service layer or repository
methods. New code should not add more unversioned FDB methods.

## Target Package Shape

The FDB core should become a bounded subsystem. The exact package move can be
phased, but the target ownership should be clear:

```text
chisurf/core/fdb/
  models.py
  schema.py
  migrations.py
  repository.py
  transactions.py
  graph.py
  archive.py
  api.py

chisurf/plugins/sample_database/backend/
  fdb_services.py
  compatibility_services.py
  ndxplorer_adapter.py
  setup_adapter.py
```

Rules:

- `chisurf/core/fdb` owns the data model, schema, migrations, transactions,
  invariants, graph traversal, and archive manifest semantics.
- Plugin service handlers translate JSON-RPC requests into core calls.
- Adapters translate workflow-specific objects into artifacts, operations,
  parameters, setup references, and edges.
- ndxplorer, Burst Selection, project archive, and fitting code should not define
  new FDB schemas directly.

## Required Invariants

The repository must enforce these invariants:

- Parent rows must not be updated with SQLite `INSERT OR REPLACE`.
- Re-saving an operation must not delete outputs, inputs, parameters, or graph
  history unless an explicit delete/versioning operation asks for it.
- Entity, link, parameter, provenance, and audit writes must be committed in one
  transaction.
- Every operation input must reference an existing artifact.
- Every operation output must reference an existing artifact.
- Every setup reference must point to an existing setup version or snapshot.
- Every parameter with `parameter_of` must reference an existing operation.
- Every public mutation must write an audit record.
- Graph traversal must be cycle-safe.
- Artifact checksums should be immutable once validated. If content changes, a
  new artifact version should be created.
- Deleting an analysis/project/archive must not leave orphan operations or
  dangling graph edges.
- Compatibility wrappers must preserve legacy behavior while writing canonical
  records.

## Migration Mapping

The current phase tables should map into the canonical model as follows:

| Current concept | Target concept |
| --- | --- |
| `fdb_raw_data` | `fdb_artifact` with `artifact_type = raw_data` |
| `fdb_processed_data` | `fdb_artifact` with typed processed artifact values |
| `fdb_processing_run` | `fdb_operation` |
| `fdb_processing_input` | `fdb_operation_artifact` with `direction = input` |
| `fdb_provenance_edge` | `fdb_operation_artifact` plus `fdb_edge` where needed |
| `fdb_analysis_run` | `fdb_operation` with `operation_type = analysis` or `fitting` |
| `fdb_analysis_parameter` | `fdb_parameter` |
| `fdb_setup_definition` | `fdb_setup` |
| project archive stored as analysis fit structure | `fdb_artifact` with `artifact_type = project_archive` plus archive operation |
| ndxplorer processing-only analysis | `fdb_operation` using the same analysis/selection taxonomy |
| archive manifest product | `fdb_artifact` with `artifact_type = archive_manifest` |

## Migration Phases

### Phase 0 - Architecture Freeze

Deliverables:

- Approve this PRD as the canonical migration target.
- Mark older phase-specific FDB PRDs as historical implementation notes.
- Add a short `docs/fdb_architecture.md` that explains the domain model without
  implementation details.
- Define the canonical artifact, operation, edge, and parameter vocabularies.

Acceptance criteria:

- Developers can answer "where do I add a new workflow?" from the architecture
  doc.
- New FDB work references this PRD or the architecture doc.

### Phase 1 - Repository Safety Foundation

Deliverables:

- Replace parent-table `INSERT OR REPLACE` writes with true UPSERT or
  update-then-insert logic.
- Add transaction helpers for unit-of-work writes.
- Make provenance and audit writes part of the same transaction as entity writes.
- Add idempotency tests for re-saving raw data, operations, artifacts, setup
  definitions, and parameters.

Acceptance criteria:

- Re-saving an operation with the same ID does not delete its products.
- Failed provenance or audit writes roll back the whole mutation.
- Tests cover the current cascade-risk failure mode.

### Phase 2 - Canonical Schema Additive Migration

Deliverables:

- Add canonical tables without deleting legacy tables.
- Add indexes for artifact type, operation type, operation status, operation
  timestamps, input/output artifact IDs, edge endpoints, setup IDs, parameter
  operation IDs, and audit target IDs.
- Add typed vocabulary validation in repository code.
- Add migration tests from the current schema version to the new version.

Acceptance criteria:

- Existing databases migrate without data loss.
- Old code paths still work.
- New canonical tables can represent all existing FDB records.

### Phase 3 - Backfill and Dual-Write

Deliverables:

- Backfill canonical artifacts from raw and processed data records.
- Backfill canonical operations from processing and analysis runs.
- Backfill operation input/output links from processing inputs and provenance
  edges.
- Backfill parameters from analysis parameter records.
- Backfill setups from setup definitions.
- Start dual-writing legacy and canonical records for compatibility methods.

Acceptance criteria:

- A migration report can count legacy rows, canonical rows, skipped rows, and
  warnings.
- Upstream/downstream graph queries return equivalent results on legacy and
  canonical paths for covered workflows.
- Backfill can be rerun idempotently.

### Phase 4 - Canonical Repository API

Deliverables:

- Add repository methods centered on artifacts, operations, links, parameters,
  setups, graph traversal, archives, and audit.
- Move workflow-specific assembly out of service handlers into adapters.
- Add cycle-safe graph traversal with visited-node tracking.
- Add explicit delete/versioning semantics.

Acceptance criteria:

- Service handlers no longer manually assemble low-level provenance rows.
- Graph traversal handles cycles without repeated edge explosion.
- Deleting analysis/project/archive records leaves no orphan canonical rows.

### Phase 5 - Versioned Service API

Deliverables:

- Add `fdb.v1.*` service methods.
- Keep legacy service methods as wrappers.
- Generate or validate the plugin manifest from the registered service catalog.
- Add API tests for request/response JSON compatibility.

Acceptance criteria:

- Every registered FDB method is discoverable in the manifest or generated API
  reference.
- New functionality appears under `fdb.v1.*`.
- Legacy callers continue to pass existing tests.

### Phase 6 - Workflow Adapter Migration

Deliverables:

- Migrate Burst Selection registration to canonical artifact/operation writes.
- Migrate ndxplorer recording to the canonical analysis/selection taxonomy.
- Migrate setup services to canonical setup snapshots.
- Migrate project archive/restore to project archive artifacts instead of fake
  fit structures.
- Migrate archive exports to canonical graph and artifact records.

Acceptance criteria:

- Each workflow uses the same artifact/operation/link model.
- ndxplorer analysis and generic analysis appear in the same operation taxonomy.
- Project archives are represented as project archive artifacts.
- Archive exports are collision-safe and checksum-verified.

### Phase 7 - Documentation and Cleanup

Deliverables:

- `docs/fdb_architecture.md`: human architecture source of truth.
- `docs/fdb_schema_reference.md`: generated or mechanically maintained schema
  reference.
- `docs/fdb_api_reference.md`: generated or mechanically validated API reference.
- Mark old phase-specific implementation details as historical.
- Remove redundant schema descriptions from broad planning docs where possible.

Acceptance criteria:

- Developers have one architecture document, one schema reference, and one API
  reference.
- No workflow doc defines a separate provenance model.
- Test names and fixtures use canonical terminology.

## Documentation Strategy

To avoid scattered schemes, documentation should be split by purpose:

```text
docs/fdb_architecture.md
  Human explanation of the model and invariants.

docs/fdb_schema_reference.md
  Table and migration reference. Prefer generated content.

docs/fdb_api_reference.md
  Service method reference. Prefer generated content.

docs/prd_fdb_architecture_migration.md
  This migration plan and acceptance criteria.
```

Older documents may remain, but they should be labeled as historical phase notes
once this PRD is accepted.

## Testing Requirements

The migration requires tests in these categories:

- schema migration from current user databases
- idempotent upsert behavior
- transaction rollback on provenance/audit failure
- graph endpoint validation
- cycle-safe upstream/downstream traversal
- legacy API wrapper compatibility
- manifest/service catalog consistency
- backfill idempotency
- archive path collision handling
- checksum validation during archive export
- project archive/restore round trip
- ndxplorer analysis taxonomy consistency
- setup snapshot validation

The minimum regression test before modifying repository writes is:

```text
create operation -> create output artifact -> save same operation ID again
expected: output artifact and provenance remain
```

## Rollout Requirements

- Every database migration must be backup-before-migration.
- The canonical schema migration must be additive first.
- Legacy tables must remain readable until canonical reads are proven equivalent.
- Dual-write should be temporary and tracked with explicit exit criteria.
- The final cleanup phase should be a separate decision after migration evidence.

## Open Questions

- Should artifact versioning be explicit in `artifact_id`, a separate
  `artifact_version`, or both?
- Should `fdb_edge` store all graph facts, or should operation input/output links
  be the only source of truth with `fdb_edge` as a view/export shape?
- Which artifact payloads are small enough to embed in SQLite?
- Which setup fields must be queryable relational columns versus setup JSON?
- What is the stable operation taxonomy for fitting versus analysis versus
  project archive?
- Should project archives contain full UI state, only scientific state, or both?
- What is the minimum archive format required for exchange outside ChiSurf?

## Success Criteria

The migration is successful when:

- A developer can add a new workflow by creating artifacts, recording one or more
  operations, linking inputs/outputs, and adding parameters/setup references.
- The same provenance graph model covers Burst Selection, ndxplorer, FCS, TCSPC,
  fitting, project archive, and archive export.
- Re-saving existing records is safe and idempotent.
- Public APIs are versioned, discoverable, and backward compatible.
- Graph queries are typed, cycle-safe, and test-covered.
- Documentation has one canonical architecture and no competing provenance
  schemes.

## Post-Implementation Review: v17 Canonical Migration

Review date: 2026-06-12

The implemented migration makes real progress:

- canonical tables were added for artifacts, operations, operation-artifact links,
  edges, parameters, setups, and audit logs
- parent FDB writes mostly moved from SQLite `INSERT OR REPLACE` to real UPSERTs
- a v17 migration/backfill path exists
- `fdb.v1.*` service aliases exist
- cycle-safe Python graph traversal helpers exist
- targeted tests cover migration, idempotent operation updates, and basic API
  calls

However, the implementation is not yet architecturally settled. It currently
introduces another database core (`mfdb`) while retaining the old mmCIF database
module, legacy FDB tables, canonical FDB tables, legacy service names, new
`mfdb.v1.*` names, and `fdb.v1.*` aliases. This keeps the project functional but
still leaves developers with multiple overlapping mental models.

### Required Improvement 1 - Choose One Public Name and One Schema Authority

Problem:

The implementation created `chisurf/core/mfdb/` while the PRD target and user
language use `fdb`. The compatibility package
`chisurf.core.fio.mmcif.db` re-exports the new repository, but the old local
`chisurf/core/fio/mmcif/db/schema.py` still exists at schema version 16. Some
imports resolve to the v17 schema and others still resolve to the v16 schema.

Required direction:

- Choose one canonical public name: preferably `fdb`.
- Keep `mfdb` only as a temporary alias if the rename has already leaked.
- Make all schema creation, migration, and backup decisions use one schema
  module.
- Stop copying the whole legacy schema into a second core module as a long-term
  architecture.

Acceptance criteria:

- `database_resolver`, repository initialization, seed data, tests, and service
  handlers all resolve the same schema module and same `SCHEMA_VERSION`.
- Creating an empty database through `resolve_database_path()` creates the
  current canonical schema, not an older intermediate schema.
- There is a deprecation note for any retained `mfdb` import path.
- There is no developer-facing ambiguity between `fdb`, `mfdb`, and
  `sample_database`.

### Required Improvement 2 - Make Repository Mutations Truly Atomic

Problem:

UPSERTs fixed the highest-risk cascade-delete class of bugs, but several methods
still commit entity rows before provenance and audit records are written. The
same issue exists in direct canonical methods: `register_artifact()` and
`record_operation()` write audit records after the main `with self.conn` block.
Other public mutations, including operation links, parameters, and setup saves,
do not write audit records at all.

Required direction:

- Add a repository unit-of-work helper and use it consistently.
- Entity writes, operation links, provenance edges, parameters, and audit logs
  must commit or roll back together.
- Audit should be part of the mutation transaction, not a best-effort follow-up.

Acceptance criteria:

- If audit insertion fails, the entity mutation rolls back.
- If provenance edge insertion fails, the entity mutation rolls back.
- Every public mutation writes one audit record unless explicitly documented as
  internal.
- Tests patch audit/provenance writes to fail and prove rollback.

### Required Improvement 3 - Decide Whether `fdb_edge` Is Source of Truth or View

Problem:

The current graph has two sources for the same operation relationship:
`fdb_operation_artifact` and `fdb_edge`. Legacy methods dual-write both, and
canonical traversal reads both. This can produce duplicate semantic edges such
as:

```text
operation -> produced -> artifact
processing_run -> produced -> processed_data
```

for the same underlying relationship.

Required direction:

- Prefer `fdb_operation_artifact` as the source of truth for operation inputs and
  outputs.
- Treat `fdb_edge` as either:
  - a compatibility/export projection, or
  - a table only for non-operation relationships such as `included_in`,
    `contains`, `uses_setup`, and `derived_from`.
- Do not store the same input/output relationship in both places unless graph
  traversal performs semantic de-duplication.

Acceptance criteria:

- A raw-data -> operation -> product traversal returns one semantic edge per
  relationship.
- Graph exports do not contain duplicate legacy/canonical synonyms.
- Tests cover dual-written legacy rows and canonical traversal output.

### Required Improvement 4 - Enforce Typed Vocabularies

Problem:

Canonical tables use free-text values for artifact types, operation types,
relationship types, storage modes, validation states, and operation-link
directions. Repository methods currently accept invalid values such as an
operation-artifact direction other than `input` or `output`.

Required direction:

- Define vocabulary constants in one module.
- Validate vocabulary values in repository methods.
- Add database `CHECK` constraints where the vocabulary is small and stable,
  especially for operation-link direction and lifecycle status.
- Keep extensible vocabularies explicit through a controlled registry or
  documented extension points.

Acceptance criteria:

- Invalid operation-link directions are rejected before insert.
- Invalid lifecycle statuses are rejected or normalized.
- Tests cover invalid artifact type, operation type, direction, edge type, and
  status.
- The architecture doc lists the canonical vocabularies and extension policy.

### Required Improvement 5 - Fix Analysis Deletion and Optional Experiment Semantics

Problem:

`add_analysis_run()` accepts `experiment_id=None`, but the legacy parent
`fdb_processing_run.experiment_id` column is `NOT NULL`, so a supposedly optional
analysis experiment can fail at insert time. Deleting an analysis still removes
only the legacy analysis child and legacy edges; the parent processing run,
canonical operation, and canonical parameters can remain behind.

Required direction:

- Decide whether every analysis must belong to an experiment.
- If yes, enforce that at the method boundary with a clear error.
- If no, the legacy parent table cannot be the required parent for canonical
  analysis records.
- Delete/version analysis records through the canonical operation model, not by
  deleting only the legacy child row.

Acceptance criteria:

- Analysis creation behavior matches its type signature and documentation.
- Deleting or superseding an analysis leaves no orphan processing runs,
  operations, operation-artifact links, parameters, or edges.
- Tests assert absence of legacy and canonical orphans after deletion.

### Required Improvement 6 - Make the Versioned API Discoverable and Complete

Problem:

The service registry registers both `mfdb.v1.*` and `fdb.v1.*`, but the plugin
manifest still advertises only the old sample database and early phase methods.
The implemented versioned API also differs from the PRD shape: it has
`graph.traverse` but not explicit `graph.upstream` / `graph.downstream`, and it
does not expose versioned archive or project methods.

Required direction:

- Generate or validate the manifest from registered methods.
- Prefer `fdb.v1.*` as the public API namespace.
- Keep `mfdb.v1.*` only if there is a clear product decision to rename the
  project.
- Add versioned wrappers for graph upstream/downstream, archives, and project
  archive/restore before encouraging new clients to use the API.

Acceptance criteria:

- Every registered `fdb.v1.*` method appears in the manifest or generated API
  reference.
- Tests fail if registered services and manifest methods drift.
- New client code can avoid unversioned legacy methods.

### Required Improvement 7 - Keep Workflow Adapters Headless

Problem:

The ndxplorer loading service still imports through a path that can pull GUI/PyQt
dependencies into a headless service call. The current FDB test suite shows this
as an import failure in `test_fdb_ndxplorer.py`.

Required direction:

- Split ndxplorer reader/data loading from GUI dependencies.
- Make the FDB ndxplorer adapter depend only on headless IO/data APIs.
- Remove runtime `sys.path` mutation from service handlers once ndxplorer is a
  proper package dependency or optional adapter.

Acceptance criteria:

- `test_fdb_ndxplorer.py` passes in a headless environment.
- Loading a registered burst product does not import PyQt.
- The adapter returns JSON-RPC-serializable values, not raw NumPy arrays.

### Required Improvement 8 - Make Backfill Auditable and Repeatable

Problem:

The v17 backfill uses `INSERT OR IGNORE`, which makes reruns idempotent but also
hides conflicts and stale canonical rows. There is no migration report showing
counts, skipped rows, conflicts, or warnings.

Required direction:

- Add a migration/backfill report object.
- Count source rows, inserted rows, updated rows, skipped rows, and conflicts per
  table.
- Log or persist warnings for malformed JSON, missing referenced rows, duplicate
  semantic edges, and type normalization.
- Decide whether rerun should update canonical rows from legacy rows or only
  insert missing rows.

Acceptance criteria:

- Migration tests assert report counts for representative legacy data.
- Backfill rerun behavior is explicit and tested.
- Conflicts are visible to developers and not silently ignored.

### Required Improvement 9 - Move Workflow Assembly Into Adapters

Problem:

The new `BurstPipeline` is useful as a proof of the canonical model, but it
hardcodes burst settings, uses status `success` while the rest of FDB commonly
uses `succeeded`, writes artifacts and links as separate operations, and creates
temporary output directories without an explicit cleanup/ownership policy.

Required direction:

- Treat Burst Selection, ndxplorer, project archive, and fitting as adapters
  around canonical repository use cases.
- Each adapter should assemble one transactionally recorded operation with
  inputs, outputs, parameters, setup references, software, status, and audit.
- Adapter status and artifact type values must use the canonical vocabulary.

Acceptance criteria:

- The burst adapter records one coherent operation from raw artifact to burst
  product.
- Failed burst execution records a failed operation without orphan artifacts.
- Temporary files have a documented ownership and archive policy.
- Status values and artifact types match the vocabulary tests.

### Recommended Next Phase - v18 Consolidation

The next implementation phase should not add more FDB feature surface. It should
consolidate the migration:

1. Pick the canonical package/API name and eliminate schema-version split-brain.
2. Make repository writes atomic with audit/provenance inside one transaction.
3. Choose the graph source of truth and remove duplicate graph semantics.
4. Add typed vocabulary validation.
5. Fix analysis delete/orphan behavior.
6. Generate or validate the service manifest.
7. Make ndxplorer adapter headless.
8. Add migration/backfill reporting.

Definition of done for v18:

- all `test/fio/test_fdb_*.py` tests pass headlessly
- schema imports resolve to one current schema version
- canonical traversal has no duplicate semantic edges for dual-written records
- failed audit/provenance writes roll back the parent mutation
- invalid graph/link vocabulary values are rejected
- analysis deletion leaves no legacy or canonical orphans
- the manifest and registered `fdb.v1.*` methods are consistent

## Educational Examples and Non-GUI API/CLI Usage

### Rebranded Multiparameter Fluorescence Database (MFDB)
The core database class has been rebranded from `FluorophoreDatabase` to `FluorescenceDatabase` (mapped to `mfdb`). For backwards compatibility, an alias `FluorophoreDatabase = FluorescenceDatabase` is maintained.

### Non-GUI API Usage via ZMQ
The ZMQ server (`ChiSurfServer`) exposes all database and plugin operations via JSON-RPC. A python client `ChisurfClient` allows connecting to the server and invoking operations programmatically without a GUI.

To simplify direct, in-process database connections and analysis execution, a high-level helper class `BurstPipeline` is provided in `chisurf.core.mfdb`. This class wraps database connection and multi-step data registration calls. The database connection is instantiated explicitly first, allowing the user to direct the pipeline to any SQLite database file:

```python
from chisurf.core.mfdb import FluorescenceDatabase, BurstPipeline

# 1. Connect to the database explicitly (optional custom DB file path)
db = FluorescenceDatabase("/path/to/database.db")

# 2. Create the pipeline, passing the database connection explicitly
pipeline = BurstPipeline(db)

# 3. Run burst selection with parameters passed as keyword arguments
pipeline.run("path/to/m000.spc", min_photons=20, time_window=1e-3)

# 4. Retrieve provenance lineage edges directly from the database
lineage = pipeline.get_lineage()
```

### Database Connection and Storage Details

#### SQLite Storage Path
By default, the Multiparameter Fluorescence Database (MFDB) stores its data in an SQLite database file.
- **Default Location**: `chisurf/core/fio/mmcif/db/sample_management.db`
- **Dynamic Selection**: You can specify a custom SQLite file by passing a path to the `FluorescenceDatabase` constructor:
  ```python
  from chisurf.core.mfdb import FluorescenceDatabase

  with FluorescenceDatabase("/path/to/custom_database.db") as db:
      # Direct Python API access (No ZMQ required)
      artifacts = db.list_artifacts()
  ```

#### ZMQ Client-Server Architecture
When working with the client-server setup (`ChiSurfServer` and `ChisurfClient`):
- **Server Instance**: The server thread boots the database connection internally, manages schema migrations on startup, and handles client requests.
- **Client Protocol**: The client connects via a standard TCP socket using JSON-RPC 2.0. This allows multiple remote clients (desktop GUI, scripts, web apps) to read and record provenance records concurrently.
- **Default Communication Ports**:
  - `cmd_port=8765`: Command and query interface (ZMQ REQ/REP socket)
  - `pub_port=8766`: Event broadcast channel (ZMQ PUB/SUB socket)

#### Manual Client-Server Setup (CLI and Python)

If you want to manage the client-server connection manually without the pipeline class helper:

##### 1. Spin up the Server
- **From the Command Line (Terminal)**:
  Run the server directly in a terminal window:
  ```bash
  python -m chisurf.server --cmd-port 8765 --pub-port 8766 --host 127.0.0.1
  ```
- **From a Python Script (Background Thread)**:
  You can run the server in a separate background thread in your script:
  ```python
  import threading
  import time
  from chisurf.server.app import ChiSurfServer

  # Initialize the server
  server = ChiSurfServer(cmd_port=8765, pub_port=8766, host="127.0.0.1")

  # Start server in background thread so it doesn't block execution
  server_thread = threading.Thread(target=server.serve_forever, daemon=True)
  server_thread.start()
  time.sleep(0.5)  # Wait briefly for startup
  ```

##### 2. Connect the Client
Once the server is running, connect your client in your script:
```python
from chisurf.core.api._client import ChisurfClient

# Connect to the running server
client = ChisurfClient(cmd_port=8765, pub_port=8766, host="127.0.0.1")
client.connect()

# Call any registered service
result = client.call("mfdb.v1.artifacts.list")
print(result)

# Clean up ZMQ socket connection when finished
client.close()
```

### CLI Usage of the Burst Selection Plugin
You can also run burst selection analysis directly from the command line using the plugin's CLI wrapper:
```bash
python -m chisurf.plugins.burst.burst_selection.cli --help
```

### Reference Examples
- **Educational Python Script**: [fdb_burst_selection_roundtrip.py](file:///Users/tpeulen/dev/chisurf/examples/fdb_burst_selection_roundtrip.py) shows a simple, flat, sequential workflow to start the server, register files, and run burst analysis using `BurstPipeline`.
- **Jupyter Notebook**: [fdb_burst_selection_roundtrip.ipynb](file:///Users/tpeulen/dev/chisurf/notebooks/fdb_burst_selection_roundtrip.ipynb) covers the same workflow and adds diagnostic plotting for intensity traces, burst sizes, burst durations, and FRET histograms.
