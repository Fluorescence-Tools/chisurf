# MFDB Implementation Review and Production-Hardening PRD

Date: 2026-06-13

Status: request changes.

This document is the current review artifact for the MFDB implementation. It
also contains the updated PRD for the next coding agent. It supersedes older
findings in this file where those findings have already been fixed.

## Scope Reviewed

- `chisurf/core/mfdb/`
- `chisurf/core/fio/mmcif/db/` compatibility shims
- `chisurf/plugins/core/mfdb_admin/`
- `chisurf/plugins/sample_database/` compatibility services
- MFDB/FDB tests under `test/fio/`
- Provenance plugin tests under `test/plugins/`
- `docs/prd_mfdb_architecture.md`

Scope correction: `flr_*` tables and PDBx/PDB-IHM/mmCIF support are valid MFDB
requirements. They are first-class scientific-domain tables, not legacy debris.
The production concern is not their existence; it is making the ownership
boundary between `flr_*`/PDBx/PDB-IHM and the `mfdb_*` provenance core explicit.

## Review Verdict

The implementation is much closer than the previous pass. The earlier Chinet
transaction bug is fixed, old `workflow_id` filtering now works, and public
relationship validation no longer accepts arbitrary values through
`add_provenance_edge`.

It is still not production-ready. The remaining blockers are narrower but
important:

1. Compound processing writes can still leave partial operation records.
2. The database schema still does not enforce the full `mfdb_edge`
   relationship vocabulary.
3. Legacy MD5 compatibility writes corrupt checksum algorithm metadata.

The architectural direction is still correct: keep MFDB relational. Do not move
the primary store to an object database. Object-like state, including Chinet
sessions, fit state, project snapshots, and GUI state, belongs in versioned
artifact payloads with queryable relational metadata around it.

## Verification Performed

### Passing tests

```bash
python -m pytest \
  test/fio/test_fdb_vocab_and_migration.py \
  test/fio/test_mfdb_chinet_adapter.py \
  test/fio/test_mfdb_chinet_fit_archive.py \
  -q
```

Result: `35 passed`.

```bash
python -m pytest \
  test/fio/test_fdb_migration_v17.py \
  test/fio/test_mmcif_flr_repository.py \
  test/plugins/test_provenance_graph_adapter.py \
  -q
```

Result: `40 passed`.

```bash
python -m pytest test/fio/test_fdb_*.py test/fio/test_mfdb_*.py -q --no-cov
```

Result: `60 passed`.

Notes:

- The two first pytest commands were run concurrently and triggered a pytest-cov
  warning about the shared `.coverage` file. The tests themselves passed.
- The broader suite was rerun with `--no-cov` to avoid the coverage-file race.
- Existing warnings remain for deprecated `FluorophoreDatabase` compatibility
  paths and a local `rocket_fft` architecture mismatch. They are not the main
  MFDB correctness blockers.

### Manual probes

The following behaviors were checked directly against a fresh MFDB database:

- `db.get_schema_version()` returns `19`.
- `add_operation(..., workflow_id="wf-old")` is retrievable through
  `get_operations(workflow_id="wf-old")`.
- `add_provenance_edge(..., relationship_type="nonsense_rel")` raises
  `ValueError`.
- Direct SQL still accepts `mfdb_edge.relationship_type = 'nonsense_rel'`.
- Chinet late parameter failure rolls back to zero operation, artifact,
  operation-artifact, and parameter rows for the attempted operation.
- `add_processing_run(..., input_raw_data_ids=["missing_raw"])` raises
  `sqlite3.IntegrityError` but leaves the operation and audit row behind.
- Legacy `add_artifact(..., md5=<md5>)` stores the digest with
  `checksum_algorithm = 'sha256'`.

## Resolved Since Previous Review

The following previous blockers are fixed or substantially addressed:

- `record_operation_with_artifacts(...)` now uses a savepoint-backed transaction
  and validates payloads before writing.
- `store_chinet_session(...)` and `archive_fit_to_mfdb(...)` now run inside
  `db.transaction()`.
- Late Chinet write failures roll back the operation, artifacts, links,
  parameters, edges, and audit rows.
- `get_operations(workflow_id=...)` now queries `metadata_json` instead of a
  nonexistent column.
- Public `add_provenance_edge(...)` validates non-operation relationship types.
- Public `add_edge(...)` rejects `input_to` and `produced`.
- v18/v19 migration converts operation input/output edges into
  `mfdb_operation_artifact` and keeps those relationships out of `mfdb_edge`.
- Fresh databases skip experimental `mfdb_sample` and `mfdb_experiment`; `flr_*`
  is the sample/experiment source of truth.

## Findings

### P0 - `add_processing_run` still leaves partial records on link failure

Evidence:

- `add_processing_run(...)` writes the operation first through
  `record_operation(...)`: `chisurf/core/mfdb/repository.py:3160`.
- It then writes input links in a later loop:
  `chisurf/core/mfdb/repository.py:3167`.
- The final audit row is also outside one explicit workflow transaction:
  `chisurf/core/mfdb/repository.py:3177`.
- The MFDB service handler calls `db.add_processing_run(...)`, then writes
  processed inputs and products in additional steps:
  `chisurf/plugins/core/mfdb_admin/backend/measurement_services.py:1070`.
- The compatibility sample-database service has the same shape:
  `chisurf/plugins/sample_database/backend/measurement_services.py:1070`.

Manual probe:

```text
processing_bad_input_error IntegrityError FOREIGN KEY constraint failed
processing_partial_counts {'operation': 1, 'links': 0, 'audit': 1}
```

Impact:

- A failed processing workflow can leave a real-looking operation with no valid
  input links.
- This is exactly the type of provenance corruption MFDB is meant to prevent.
- Tests pass because they do not cover this failure path.

Required fix:

- Make processing-run persistence one unit of work.
- Either implement `add_processing_run_with_artifacts(...)` using
  `record_operation_with_artifacts(...)`, or refactor `add_processing_run(...)`
  so operation creation, input links, output products, and audit are inside one
  `with self._transaction():` block.
- Service handlers must not assemble multi-step workflow writes outside a
  transaction. They should call one high-level repository method or explicitly
  open `with db.transaction():` around the whole workflow.

Acceptance tests:

- Missing raw input artifact leaves no `mfdb_operation`, no
  `mfdb_operation_artifact`, no `mfdb_audit_log`, and no partial output artifacts
  for the attempted processing ID.
- Missing processed input artifact in
  `mfdb.v1.processing.record` / compatibility service leaves zero partial rows.
- Simulated product-write failure after operation creation rolls back the whole
  processing workflow.
- Simulated audit failure rolls back the whole processing workflow.

### P1 - `mfdb_edge` does not enforce the full relationship vocabulary at the DB boundary

Evidence:

- Fresh schema only prevents `input_to` and `produced` in `mfdb_edge`:
  `chisurf/core/mfdb/schema.py:825`.
- Built-in allowed relationship values are defined in vocabulary bootstrap:
  `chisurf/core/mfdb/schema.py:1015`.
- Repository APIs validate relationship types, but direct SQL can still insert
  arbitrary values.

Manual probe:

```text
api_invalid_rel_rejected ValueError Invalid relationship_type 'nonsense_rel'
direct_invalid_rel_accepted
```

Impact:

- The graph invariant depends on repository discipline rather than the database.
- Any migration, repair script, plugin, or future direct SQL utility can create
  graph relationships that query code cannot reason about.
- This undermines the PRD requirement that MFDB be future-proof and auditable.

Required fix:

- Keep the repository validation, but add DB-level enforcement too.
- Because `relationship_type` is extensible through `mfdb_vocabulary`, prefer
  SQLite triggers over a static `CHECK` list:
  - `BEFORE INSERT ON mfdb_edge`
  - `BEFORE UPDATE OF relationship_type ON mfdb_edge`
  - reject `input_to` and `produced`
  - reject values for which no active row exists in `mfdb_vocabulary` with
    `field_name = 'relationship_type'`
- Add a schema migration, likely v20, that:
  - scans existing invalid `mfdb_edge` relationship values
  - reports them in `MigrationReport`
  - either maps known legacy aliases or aborts with a clear repair message
  - installs the triggers after cleanup

Acceptance tests:

- Fresh DB direct SQL rejects `input_to`, `produced`, and `nonsense_rel`.
- Fresh DB direct SQL accepts seeded active values such as `contains` and
  `parameter_depends_on`.
- Fresh DB direct SQL accepts a newly registered active relationship value.
- Fresh DB direct SQL rejects the same value after it is marked inactive.
- Migrated DB has the same trigger behavior.
- Migration reports invalid legacy edge rows instead of silently preserving them.

### P1 - Legacy MD5 compatibility stores incorrect checksum metadata

Evidence:

- `add_artifact(...)` accepts a legacy `md5` argument and forwards it as
  canonical `checksum`: `chisurf/core/mfdb/repository.py:670`.
- It does not pass `checksum_algorithm="md5"`, so `register_artifact(...)`
  applies its default `sha256`: `chisurf/core/mfdb/repository.py:2233`.

Manual probe:

```text
legacy_md5_checksum_algorithm sha256
```

Impact:

- A stored checksum can be cryptographically and semantically mislabeled.
- Downstream validation can recompute SHA-256, compare it to an MD5 digest, and
  incorrectly mark valid artifacts invalid.
- Scientific archive verification becomes unreliable.

Required fix:

- If `md5` is provided to the compatibility shim, pass
  `checksum_algorithm="md5"`.
- Prefer introducing an explicit `checksum` plus `checksum_algorithm` path and
  deprecating `md5` in the public compatibility method.
- Add checksum format validation where feasible:
  - MD5: 32 hex characters
  - SHA-256: 64 hex characters
- Do not default to `sha256` when a checksum value is known to come from an
  algorithm-specific legacy argument.

Acceptance tests:

- `add_artifact(..., md5=<32 hex>)` stores `checksum_algorithm = 'md5'`.
- `register_artifact(..., checksum=<64 hex>)` defaults to `sha256`.
- Invalid checksum length for the declared algorithm raises before write.
- Migration from legacy tables preserves `md5` as MD5 metadata or maps it to
  canonical checksum with `checksum_algorithm = 'md5'`.

### P2 - Canonical and compatibility responsibilities remain too mixed

Evidence:

- `MFDatabase` contains canonical methods, old FDB compatibility methods, FLR
  accessors, analysis-run compatibility methods, migration helpers, graph
  helpers, and archive helpers in one large class.
- Compatibility services still emit many deprecation warnings while tests pass.
- Service handlers still compose workflow writes manually instead of consistently
  calling the high-level atomic API.

Impact:

- The architecture is hard for a new developer to understand.
- Future coders are likely to add another compatibility path rather than use the
  canonical model.
- Test coverage can pass while important production invariants are violated.

Required fix:

- Split responsibility without changing storage:
  - `MFDatabase`: canonical core repository only.
  - `FlrDomainRepository` or clearly marked methods: FLR/PDBx/PDB-IHM domain
    access through the same connection.
  - `LegacyFDBAdapter`: compatibility method names such as old processing,
    analysis, and sample database calls.
  - `MFDBWorkflowWriter`: high-level atomic workflow entrypoints used by plugins.
- Compatibility wrappers may remain, but they must be thin and visibly separated.
- New service code should import the canonical module, not old mmCIF/FDB paths.

Acceptance tests:

- Public canonical methods are listed in one registry or protocol.
- Compatibility methods are tested separately and marked deprecated.
- Repo search test prevents new canonical code from importing old MFDB authority
  paths except in compatibility modules.

## Updated Production-Hardening PRD

### Objective

Make MFDB, the multiparameteric fluorescence database, production-ready as a
relational SQLite provenance and archive database for fluorescence workflows.
The next implementation pass must remove the remaining corruption paths and make
the architecture understandable to a developer who did not work on FDB.

### Non-Goals

- Do not replace SQLite with an object database.
- Do not remove `flr_*`, PDBx, PDB-IHM, or mmCIF support.
- Do not reintroduce persisted `input_to` or `produced` rows in `mfdb_edge`.
- Do not keep adding new ad hoc low-level service writes.
- Do not preserve pre-production FDB quirks for their own sake; breaking changes
  are allowed before production.

### Core Architecture

MFDB has three layers:

1. Scientific domain layer:
   - `flr_*`
   - PDBx/PDB-IHM/mmCIF-compatible tables
   - sample, experiment, probe, structure, and fluorescence-domain metadata

2. Provenance core:
   - `mfdb_artifact`
   - `mfdb_operation`
   - `mfdb_operation_artifact`
   - `mfdb_parameter`
   - `mfdb_edge`
   - `mfdb_setup`
   - `mfdb_audit_log`
   - `mfdb_vocabulary`

3. Compatibility and adapter layer:
   - old sample-database/FDB method names
   - GUI-facing RPC wrappers
   - import/export adapters
   - Chinet, ndxplorer, Burst Selection, project archive writers

Only the provenance core owns workflow lineage. The domain layer owns
fluorescence science entities. Compatibility code must translate into these two
layers; it must not create a second model.

### Required Invariants

- Every public mutation is atomic.
- Every audit row is committed or rolled back with the mutation it describes.
- Operation input/output links are stored only in `mfdb_operation_artifact`.
- `mfdb_edge` stores only non-operation relationships.
- `mfdb_edge.relationship_type` is enforced by active `mfdb_vocabulary` rows at
  the database boundary.
- Unknown or inactive vocabulary values are rejected by repository methods and
  by direct SQL where practical.
- Legacy checksum fields must not lie about the checksum algorithm.
- Fresh and migrated databases enforce the same invariants.
- Compatibility wrappers must call canonical methods or one high-level workflow
  writer.
- Full object state is stored as versioned artifact payloads, never as the
  primary database model.

### Required Implementation Work

#### 1. Add a canonical workflow writer

Implement or formalize one high-level writer for operation-centered workflows.
It can be an `MFDatabase` method or a small helper class, but all adapters must
use it.

Minimum API:

```python
record_operation_with_artifacts(
    operation_id: str,
    operation_type: str,
    status: str = "pending",
    experiment_id: str | None = None,
    setup_id: str | None = None,
    settings: dict | None = None,
    input_artifacts: list[ArtifactPayload] | None = None,
    output_artifacts: list[ArtifactPayload] | None = None,
    parameters: list[ParameterPayload] | None = None,
    edges: list[NonOperationEdgePayload] | None = None,
    metadata: dict | None = None,
) -> OperationWriteResult
```

Rules:

- Validate every payload before the first write.
- Register missing declared output artifacts inside the transaction.
- Input artifacts may either be pre-existing IDs or full payloads, but missing
  IDs must fail before partial writes remain.
- Record all operation-artifact links inside the same transaction.
- Record parameters inside the same transaction.
- Record non-operation edges inside the same transaction.
- Record audit inside the same transaction.
- Return actual inserted/updated/skipped counts.

#### 2. Refactor processing and analysis services to use the workflow writer

Affected entrypoints:

- `MFDatabase.add_processing_run`
- `MFDatabase.add_processed_data_product`
- `MFDatabase.add_analysis_run`
- `MFDatabase.add_analysis_product`
- `mfdb.v1.processing.record` or equivalent service handlers
- old sample-database `record_processing_run_handler`
- old sample-database `record_analysis_run_handler`
- ndxplorer persistence services
- project archive services

Rules:

- No service handler may create an operation, then later link artifacts, then
  later add products without one transaction around all steps.
- If a service needs compatibility-shaped inputs, normalize them into canonical
  payloads first, then call the workflow writer.
- Unknown keyword arguments in repository methods must either be consumed,
  stored in metadata, or rejected. Silent discard is not acceptable for
  production API methods.

#### 3. Add DB-level relationship vocabulary enforcement

Implement migration v20:

- Add triggers for `mfdb_edge` insert/update.
- Reject `input_to` and `produced`.
- Reject relationship values not present as active
  `mfdb_vocabulary(field_name='relationship_type', value=...)`.
- Keep extensibility by allowing `register_vocabulary_value(...)` to add active
  values before edge insertion.

Migration behavior:

- Existing invalid rows must be reported.
- Known aliases may be mapped in a documented table.
- Unknown invalid values should abort migration with a repair message unless the
  project explicitly chooses quarantine.

#### 4. Fix checksum compatibility

Requirements:

- `add_artifact(..., md5=...)` writes `checksum_algorithm='md5'`.
- `register_artifact(...)` keeps `sha256` as the default only when callers use
  the algorithm-neutral `checksum` argument.
- Add optional checksum-format validation.
- Document that `metadata_json.md5` is compatibility metadata only; canonical
  validation must use `checksum` and `checksum_algorithm`.

#### 5. Clarify module boundaries

Target structure:

```text
chisurf/core/mfdb/
  repository.py          canonical repository and connection lifecycle
  schema.py              schema creation, migrations, vocabulary bootstrap
  transactions.py        savepoint-backed unit-of-work helper
  graph.py               read-only graph traversal and graph export
  workflows.py           high-level atomic workflow writer
  flr_domain.py          FLR/PDBx/PDB-IHM domain access helpers if split out
  compatibility.py       old FDB/sample_database method adapters
  chinet_adapter.py      Chinet-specific adapter using workflows.py
```

This split can be staged. The immediate requirement is that new implementation
work must not make `repository.py` more monolithic.

#### 6. Update docs after implementation

Update `docs/prd_mfdb_architecture.md` after the code changes:

- State that `flr_*` and PDBx/PDB-IHM are first-class domain tables.
- State that `mfdb_*` is the provenance core.
- State that compatibility wrappers are non-authoritative.
- Document the final relationship vocabulary enforcement mechanism.
- Document the high-level workflow writer as the only recommended adapter write
  path.

## Required Tests for Next Pass

### Transaction tests

- `add_processing_run` with missing raw input rolls back operation and audit.
- Processing service with missing processed input returns an error and leaves no
  partial operation.
- Processing service with product write failure leaves no partial operation.
- Analysis service with parameter write failure leaves no partial operation,
  products, parameters, or edges.
- Project archive failure after operation creation leaves no partial operation.
- Existing Chinet rollback tests continue to pass.

### Schema tests

- Fresh DB direct SQL rejects `input_to`, `produced`, and `nonsense_rel` in
  `mfdb_edge`.
- Fresh DB direct SQL accepts every active built-in relationship.
- Fresh DB direct SQL accepts a newly registered active relationship.
- Fresh DB direct SQL rejects an inactive relationship.
- Migrated DB has the same behavior.
- Migration report includes invalid edge quarantine/abort details.

### Checksum tests

- Legacy `md5` stores canonical algorithm `md5`.
- Canonical SHA-256 stores canonical algorithm `sha256`.
- Wrong digest length for declared algorithm raises before write.
- Imported legacy MD5 data is not mislabeled as SHA-256.

### API boundary tests

- Every public canonical repository method has a smoke test.
- Unknown kwargs to canonical methods raise.
- Compatibility methods are listed separately and marked deprecated.
- Plugin manifest methods exactly match registered `mfdb.v1.*` methods.
- No canonical service registers `fdb.v1.*`.

## Implementation Order

1. Fix `add_processing_run` transactionality and add rollback tests.
2. Refactor service processing/analysis writes to use one transaction or the
   workflow writer.
3. Fix legacy MD5 checksum algorithm handling and tests.
4. Add relationship-vocabulary triggers and migration v20.
5. Add schema/direct-SQL tests for fresh and migrated databases.
6. Split or clearly quarantine compatibility code from canonical repository
   code.
7. Update `docs/prd_mfdb_architecture.md` to match final code.

## Merge Gate

Do not call the implementation done until all of the following are true:

- The manual `add_processing_run` missing-input probe leaves zero partial rows.
- Direct SQL cannot insert invalid `mfdb_edge.relationship_type` values.
- Legacy MD5 writes are labeled as MD5.
- Focused MFDB/FDB tests pass.
- Chinet rollback tests pass.
- The broader `test/fio/test_fdb_*.py test/fio/test_mfdb_*.py` suite passes.
- The PRD and implementation agree on `flr_*`, PDBx/PDB-IHM, and `mfdb_*`
  responsibilities.

