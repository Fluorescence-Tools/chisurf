---
type: PRD
prd: "03"
title: "PRD-03: Result Registry"
description: A single register_result() API so any plugin can archive output to MFDB with full provenance
status: in-progress
phase: "0"
resource: chisurf/core/mfdb/result_registry.py
tags: [prd, mfdb, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Provides a single `register_result()` function any plugin can call to archive its
output in MFDB with full provenance — object stored, artifact and operation rows
created, input/output and derived-from / measured-sample edges wired, and
parameters recorded — without the plugin needing to understand MFDB internals.
Most plugins produce data but do not write to MFDB; this gives them a
dead-simple API and one well-tested choke point so provenance is uniform. It is
the lightweight per-plugin path that complements the whole-project archiver,
both calling the same repository primitives, and it serializes exclusively
through the payload codecs rather than ad-hoc JSON.

# Status
In progress. Blocked on and built against the payload codec layer; the verified
repository API signatures are pinned below, and a code review is on record (see
Review outcomes).

# Goal
Create a single function that any plugin can call to register its output in MFDB.
Plugins should not need to understand MFDB internals — just call
`register_result()` and the data is archived with full provenance (object stored,
artifact + operation rows created, input/output and `derived_from` /
`measured_sample` edges wired, parameters recorded). At the time of writing, 79 of
87 plugins produce data but do not write to MFDB; this PRD gives them a
dead-simple API and a single, well-tested choke point so provenance is uniform.

# Blocking dependency
The result registry stores and retrieves scientific payloads (FCS curves,
spectra, TTTR, TCSPC decays, burst tables, …). Those payload formats must be
formally defined first by the payload-formats/codecs PRD, which establishes the
typed, msgpack-based payload schemas and codec registry. Until then,
`register_result()` can only serialize data ad-hoc (lossy JSON), the exact
weakness this design is meant to remove. PRD-03's `_store_data()` and any
`read_result()` must use those codecs rather than `to_json()` / `json.dumps()`.

# Verified repository API
The registry is written against the real `MFDatabase` method signatures in
`chisurf/core/mfdb/repository.py` (an earlier draft used invented method names —
these are the actual ones):

| Purpose | Real method | Key arguments |
|---------|-------------|---------------|
| Store bytes/file in object store | `db.put_object(path=None, data=None, filename=None, mime_type=None, metadata=None)` | Returns a **dict** `{"object_uuid", "content_md5", "size_bytes", "original_filename", "deduplicated", "storage_path", "refcount"}` |
| Create/update artifact row | `db.register_artifact(artifact_id, artifact_kind=, data_format=, storage_mode=, object_uuid=, size_bytes=, mime_type=, checksum=, metadata=)` | `metadata=` is a **dict** (auto-serialized), not `metadata_json`. `storage_mode` must be in `STORAGE_MODES`. |
| Create/update operation row | `db.record_operation(operation_id, operation_type, status="pending", settings=None, metadata=None, software_module=None, started_at=None, ended_at=None)` | `operation_type` is **extensible** (validated via `validate_extensible_vocab`). `status` must be in `STATUS_VALUES` (use `"succeeded"`). |
| Link operation ↔ artifact | `db.record_operation_link(operation_id, artifact_id, direction, role=None, ordinal=0)` | `direction` ∈ `{"input","output"}`. |
| Record a parameter | `db.record_parameter(parameter_uuid, operation_id, name, value=None, standard_error=None, lower_bound=None, upper_bound=None, units=None, parameter_type="free")` | Needs a **`parameter_uuid`** (generate `uuid4`). Columns are `name`/`value`, **not** `parameter_name`/`parameter_value`. `parameter_type` ∈ `PARAMETER_TYPES`. |
| Generic graph edge | `db.add_edge(source_node_type, source_node_id, target_node_type, target_node_id, relationship_type)` | `relationship_type` must be in `RELATIONSHIP_TYPES`. Cannot be `input_to`/`produced` (those go through `record_operation_link`). |
| Connection handle | `db.conn` (property) | There is **no** `db.con`. |
| Transactions | Every `record_*`/`register_*`/`add_*` wraps itself in `self._transaction()` and commits. | **Do not** call `db.conn.commit()` yourself. |

Relevant vocabulary (from `models.py`):
- `ARTIFACT_KINDS`: `raw_measurement`, `processed_data`, `analysis_result`,
  `fit_result`, `fcs_correlation`, `irf_curve`, `spectra`, `burst_table`,
  `tcspc_decay`, `pda_histogram`, `anisotropy_curve`, `selection_mask`,
  `visualization`, `external_reference`, plus legacy aliases. **Extensible** —
  unknown values are accepted and auto-registered by `validate_extensible_vocab`.
- `OPERATION_TYPES`: `measurement_import`, `burst_selection`, `fcs_correlation`,
  `tcspc_fitting`, `model_fitting`, `tcspc_histogram_computation`,
  `pda_histogram_computation`, `pch_histogram_computation`, `local_fit`,
  `global_fit`, `analysis`, plus legacy. **Also extensible.**
- `STORAGE_MODES`: `local_file`, `embedded_json`, `embedded_blob`, `url`, …
- `RELATIONSHIP_TYPES`: `derived_from`, `measured_sample`, `supersedes`,
  `linked_to`, …
- `STATUS_VALUES`: `pending`, `running`, `succeeded`, `failed`, …

# Design and tasks

## Task 1 — the result registry module
Create `chisurf/core/mfdb/result_registry.py` with the public entry point
`register_result(kind, data=None, sample_id="", parent_artifact_id="",
operation_type="", parameters=None, metadata=None, data_format="", db=None) -> str`.
It returns the created `artifact_id` (or `""` if no DB). Behavior:

1. Resolve the DB (`_get_global_db()` if `db is None`); if still none, log a
   warning and return `""` — **never raise into a plugin's normal flow**.
2. Store the payload (if any) in the object store via `_store_data`.
3. Create the artifact row (`register_artifact`).
4. Create the operation row (`record_operation`, `status="succeeded"`,
   `operation_type` defaulting to `"analysis"`).
5. Link operation → artifact as `output`.
6. If `parent_artifact_id`: add an `input` operation link to the parent **and** a
   `derived_from` edge (new artifact → parent).
7. If `sample_id`: create a `measured_sample` edge via
   `sample_manager.link_artifact_to_sample`.
8. Record parameters via `_record_parameters`.

The whole body is wrapped so any exception is logged and returns `""`.

`data` handling in `_store_data`: `str`/`Path` → file copied into the object
store (`storage_mode="local_file"`, format inferred from suffix); `bytes` →
stored as-is (`embedded_blob`); `dict`/`DataFrame` → serialized to JSON bytes
(`embedded_json`, `application/json`); `None` → metadata-only artifact.
**Once the codecs land, the per-type JSON branches are replaced by a call into
the codec registry (`encode_payload(kind, data)`, msgpack-based) so payloads
round-trip losslessly.**

`_record_parameters` writes each `name -> value` (or `name -> {value, error,
fixed, bounds, units}`) as an `mfdb_parameter` row with a fresh `parameter_uuid`,
mapping `fixed` to `parameter_type` `fixed`/`free`.

## Task 2 — `_get_global_db()`
Layered resolver so plugins need not pass a DB:
1. An explicitly set global DB (`set_global_db()` — for tests / headless).
2. The `DatabaseConnector` singleton's open handle
   (`chisurf/plugins/core/database_connector/services.py` `_connector._db`) if any.
3. A fresh `MFDatabase(resolve_database_path())`.

Returns `None` only if even opening the user DB fails (truly headless). Option 3
opens a fresh handle per call — safe (thin SQLite wrapper, auto-commit) but not
free; hot loops should open one `MFDatabase` and pass `db=`, or call
`set_global_db()` once.

## Task 3 — reconcile artifact kinds and operation types (do not blindly add)
Reuse canonical names; add only genuinely new vocabulary. Map draft proposals:
`correlation_data`→`fcs_correlation`, `irf_data`→`irf_curve`,
`spectrum_data`→`spectra`, `burst_data`→`burst_table` (all reuse). Genuinely new:
`trace_data`, `image_data`, `background_data`. New operation types where missing:
`background_correction`, `image_analysis`, `population_selection` (reuse
`tcspc_histogram_computation` etc. where specific). Both vocabularies are
extensible, so introduce new values either by adding to the tuple in `models.py`
(preferred for first-class values, plus any `schema.py` CHECK constraint) or at
runtime via `db.register_vocabulary_value(...)`. Document which values were added
and where; never remove or rename existing values.

## Task 4 — convenience wrappers
Append thin wrappers over `register_result`:
- `register_raw_measurement(file_path, sample_id, metadata, db)` — kind
  `raw_measurement`, operation `measurement_import`.
- `register_processed_data(data, parent_artifact_id, …)` — kind `processed_data`.
- `register_fit_result(fit_data, parent_artifact_id, …)` — kind `fit_result`,
  operation `local_fit`.
- `register_calibration(data, calibration_type, sample_id, parent_artifact_id,
  parameters, method, notes, db)` — kind `calibration_data`, operation
  `calibration`. `method="user_provided"` with empty parent records a
  literature/prior value; `notes` can carry a citation. Downstream calibration
  work builds directly on this wrapper — keep its signature stable.
  `register_calibration` introduces `calibration_data` and `calibration`; add both
  in Task 3.

## Task 5 — tests
`test/fio/test_result_registry.py` uses the real fixture pattern: a plain
`MFDatabase(<tmp>/test.db)` (object store root resolved internally — do not pass
`object_store_root=` or pre-create an objects dir). Query via `db.conn`; read
`mfdb_parameter` columns `name`/`value`. Cases: dict data, file data
(format-from-suffix), `derived_from` edge, input operation link, sample linkage,
parameters (dict + scalar), object-store content dedup, metadata-only artifact,
no-DB returns `""` without raising, and the convenience wrappers.

## Task 6 — one reference integration
Wire one high-value, low-risk producer (recommended: the FCS correlator or the
burst-selection output) so the rest of the plugin fleet has a worked example.
After it computes its result, best-effort call `register_result(kind=
"fcs_correlation", data=…, operation_type="fcs_correlation", parameters=…,
metadata=…)` in a `try/except pass` — MFDB is optional and must never break the
plugin. This proves the payload archive path (object store + artifact + operation
+ output link). Full sample/source provenance only follows once the plugin can
pass a real `sample_id`/`parent_artifact_id`; until those contracts exist the FCS
reference integration is payload-only provenance.

# Non-goals / boundaries
- Not a replacement for `project_archiver.py` (whole-project archival). Both share
  repository primitives; this is the lightweight per-plugin path.
- Not responsible for sample creation. Callers pass an existing `sample_id`;
  sample definition belongs to the sample-management PRD.
- No new threading model. SQLite + auto-commit is the existing contract; hot loops
  pass an explicit `db=`.

# Definition of Done
- Payload codec registry exists; `_store_data()` and the read path use the codecs
  (msgpack), not `to_json()`/`json.dumps()`.
- `result_registry.py` exists with `register_result()` + 4 convenience wrappers.
- All calls use the verified repository methods with correct argument names; no
  invented methods, no `db.con`, no manual `commit()`.
- `_get_global_db()` resolves explicit override → connector handle → fresh user
  DB; `set_global_db()` override exists.
- Object payloads stored via `put_object` with content-addressed dedup working.
- Provenance correct: `derived_from` edge + `input` operation link for parents;
  `measured_sample` edge for samples; `output` link for the result.
- Parameters written to `mfdb_parameter` with `name`/`value` columns.
- Vocabulary reconciled (Task 3): existing names reused; genuinely new values
  added to `models.py` and any `schema.py` CHECK lists.
- `register_result` never raises into plugin flow; returns `""` when no DB.
- One real plugin wired as the reference integration.
- All tests in `test/fio/test_result_registry.py` pass.

# Review outcomes
A focused code review of the registry + payload-codec scope **approved** the
implementation with no blocking or non-blocking defects. Durable conclusions:
- Typed payload-kind matching is enforced: `BurstSelection.mask` DataFrame columns
  use explicit boolean parsing (not NumPy string truthiness); ambiguous mask
  values (e.g. `"maybe"`) are rejected atomically, leaving zero artifact/object
  rows. String masks like `["False","True","0","1","no","yes"]` round-trip to the
  correct booleans; scalar spectrum `normalized="False"` round-trips as `False`.
- Unsupported known-kind DataFrames fail atomically (no partial rows). FCS
  DataFrame registration preserves both `artifact_kind` and payload kind as
  `fcs_correlation`; FCS shape is validated.
- The FCS correlator integration is correctly documented as payload-only
  provenance until the plugin can pass real `sample_id`/`parent_artifact_id` — a
  downstream integration-contract concern, not a registry/codec blocker.
- Verification: the payload-codec and result-registry test suites pass together
  (61 passed).

# Relationships
- Blocked by / uses codecs from [PRD-030](prd-030.md); links results to samples from [PRD-02](prd-02.md).
- Reference consumer: the burst pipeline [PRD-04](prd-04.md); downstream calibration [PRD-05](prd-05.md).
- Adds a uniform write path over the [MFDB (current)](/architecture/mfdb.md) store for [plugins](/architecture/plugin-system.md); relates to [Plugins target](/specs/plugins.md).
