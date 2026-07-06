---
type: PRD
prd: "01"
title: "PRD-01: Fix MFDB Project Round-Trip"
description: Make archiving a project to MFDB and restoring it produce an identical project
status: planned
phase: "0"
resource: modules/mfdb/src/mfdb/project_archiver.py
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Archiving a project to MFDB and then restoring it must reproduce the original
project exactly. Today the round-trip is lossy: only the inner fit-state payload
is stored (dropping fit id, name, model name, plot state, and range), multiple
datasets collapse to a single key, project-level metadata is not persisted, and
chinet sessions, parameters, and dependency edges are discarded on restore. This
PRD stores full fit records plus project metadata during archival and reads back
fit groups, all datasets, parameters, and edges on restore, backed by a
round-trip regression test.

# Status
Planned. The PRD enumerates concrete fixes in `project_archiver.py` and the
project-browser restore handler, with a new round-trip test as the gate.

# Goal
`archive_project_to_mfdb()` followed by `restore_project_from_artifacts()` must
produce a project identical to the original. Currently it doesn't — fit
structure is lost, datasets can vanish, chinet sessions are discarded.

# Background
Read these files before starting:
- `project_archiver.py` — the archiver and restorer (in the mfdb package)
- `chisurf/core/project/fit_state.py` — fit serialization
- `chisurf/core/project/project.py` — Project dataclass
- `chisurf/plugins/core/project_browser/backend/services.py` — restore handler

# Tasks

## Task 1: Store the full fit record (not just `fit_state_payload`)
`_archive_fits()` (around line 290) stores only the inner `fit_state_payload`
dict. The fit-group envelope is lost: `id` (fit UID), `name` (user-visible
name), `model_name` (e.g. "FRET: FD (Gaussian)"), `plot_state` (axis ranges,
log scale, visible curves), `fit_range` (data range for chi-squared), and the
`local_fits` list structure.

Fix: when creating the `fit_result` artifact, include the full fit record (as
returned by `make_fit_record()` in `fit_state.py`) in the artifact data, and put
a summary in `metadata_json`:

```python
metadata = {
    "fit_id": fit_record["id"],
    "fit_name": fit_record.get("name", ""),
    "model_name": fit_record.get("model_name", ""),
    "fit_range": fit_record.get("fit_range"),
    "plot_state": fit_record.get("plot_state"),
    "local_fits_count": len(fit_record.get("local_fits", [])),
}
```

The artifact data stored in the object store should be the full `fit_record`
dict as JSON.

## Task 2: Store project-level metadata
In `archive_project_to_mfdb()` the metadata dict (around line 162) only has
`project_id`, `version_number`, `fit_count`, `dataset_count`. Add
`chisurf_version`, `project_format_version`, `description`, `created`,
`ui_state`, and `experiments`:

```python
metadata={
    # ... existing fields ...
    "chisurf_version": payload.get("chisurf_version", ""),
    "project_format_version": payload.get("project_format_version", ""),
    "description": payload.get("description", ""),
    "created": payload.get("created", ""),
    "ui_state": payload.get("ui_state", {}),
    "experiments": payload.get("experiments", {}),
}
```

## Task 3: Fix dataset restore
Two bugs in `restore_project_from_artifacts()` (around line 580):
- It filters by `kind == "processed_data" and role == "dataset"`, but `role`
  comes from the `mfdb_operation_artifact` junction table. If the query doesn't
  JOIN properly, `role` is always empty and ALL datasets are silently dropped.
- The fallback `ds_id = ds_id.get("ds_id", role)` means all datasets get key
  `"dataset"`, so only the last one survives.

Fix:
1. Verify `db.get_operation_artifacts()` (in `repository.py`) returns rows with
   a `role` column from `mfdb_operation_artifact`; fix the query to JOIN and
   include `role` if not.
2. Change the dataset-key logic:
   ```python
   meta = art.get("metadata_json") or {}
   if isinstance(meta, str):
       meta = _json_loads(meta) or {}
   ds_id = meta.get("ds_id") or meta.get("dataset_uid") or art.get("artifact_id") or str(uuid.uuid4())
   ```
3. In `_archive_datasets()`, store `metadata_json` containing `ds_id`,
   `filename`, and `reader_class` per dataset.

## Task 4: Restore fit records with full structure
`restore_project_from_artifacts()` returns a flat list of fit_state dicts, but
the project loader expects fit-group records with `id`, `name`, `model_name`,
`local_fits`. Read fit_result artifacts as full fit records (Task 1 now stores
them that way):

```python
fit_data = _json_loads(artifact_data_bytes)
if isinstance(fit_data, dict) and "local_fits" in fit_data:
    fits.append(fit_data)
else:  # legacy fallback: wrap in a fit-record structure
    meta = art.get("metadata_json") or {}
    if isinstance(meta, str):
        meta = _json_loads(meta) or {}
    fits.append({
        "id": meta.get("fit_id", str(uuid.uuid4())),
        "name": meta.get("fit_name", "Restored Fit"),
        "model_name": meta.get("model_name", ""),
        "local_fits": [fit_data] if fit_data else [],
    })
```

## Task 5: Use chinet sessions on restore
`restore_project_handler()` in
`chisurf/plugins/core/project_browser/backend/services.py` (around line 387)
discards the `chinet_sessions` returned by `restore_project_from_artifacts()`.
Pass them (plus `experiments` and `ui_state`) into the restored payload so the
project loader can re-establish parameter dependencies:

```python
payload = {
    "datasets": restored["datasets"],
    "fits": restored["fits"],
    "experiments": restored.get("experiments", {}),
    "ui_state": restored.get("ui_state", {}),
    "chinet_sessions": restored.get("chinet_sessions", []),
}
```

## Task 6: Query parameters and edges on restore
`restore_project_from_artifacts()` never reads `mfdb_parameter` or `mfdb_edge`.
After collecting fits, query parameters per fit operation and the dependency
edges, and return them alongside `ui_state`/`experiments` from the project
operation's `metadata_json`:

```python
all_parameters = {}
for fit_op_id in fit_operation_ids:
    rows = db.con.execute(
        "SELECT * FROM mfdb_parameter WHERE operation_id = ?", (fit_op_id,)
    ).fetchall()
    all_parameters[fit_op_id] = [dict(r) for r in rows]

dependency_edges = [dict(r) for r in db.con.execute(
    "SELECT * FROM mfdb_edge WHERE relationship_type = 'parameter_depends_on'"
).fetchall()]

return {
    "datasets": datasets,
    "fits": fits,
    "chinet_sessions": chinet_sessions,
    "parameters": all_parameters,
    "dependency_edges": dependency_edges,
    "ui_state": project_metadata.get("ui_state", {}),
    "experiments": project_metadata.get("experiments", {}),
}
```

Collect `fit_operation_ids` while iterating fits, and read `project_metadata`
from the project operation's `metadata_json`.

## Task 7: Write a round-trip test
Create `test/fio/test_mfdb_project_roundtrip.py` with a temporary MFDB fixture
and a minimal payload (2 datasets, 2 fits, `ui_state`, `experiments`). The key
assertions:
1. Two datasets in → two datasets out (not collapsed to a single key).
2. Fit records have `id`, `name`, `model_name`, `local_fits`.
3. UI state survives the round-trip.
4. Dataset UIDs are preserved.

# Definition of Done
- [ ] `archive_project_to_mfdb()` stores full fit records with id/name/model_name
- [ ] `archive_project_to_mfdb()` stores project-level metadata (version, ui_state, experiments)
- [ ] `restore_project_from_artifacts()` returns fit-group records, not flat dicts
- [ ] `restore_project_from_artifacts()` returns all datasets with unique keys
- [ ] `restore_project_from_artifacts()` returns parameters and dependency edges
- [ ] Restore handler in project_browser uses chinet sessions
- [ ] Round-trip test passes

# Relationships
- Foundational fix that later result/provenance PRDs build on: [PRD-030](prd-030.md), [PRD-03](prd-03.md).
- Operates on the [MFDB (current)](/architecture/mfdb.md) store toward its [MFDB target](/specs/mfdb.md).
