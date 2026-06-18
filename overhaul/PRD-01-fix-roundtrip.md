# PRD-01: Fix MFDB Project Round-Trip

## Goal

`archive_project_to_mfdb()` followed by `restore_project_from_artifacts()` must produce
a project identical to the original. Currently it doesn't -- fit structure is lost,
datasets can vanish, chinet sessions are discarded.

## Background

Read these files before starting:
- `chisurf/core/mfdb/project_archiver.py` -- the archiver and restorer
- `chisurf/core/project/fit_state.py` -- fit serialization
- `chisurf/core/project/project.py` -- Project dataclass
- `chisurf/plugins/core/project_browser/backend/services.py` -- restore handler

## Tasks

### Task 1: Store Full Fit Record (not just fit_state_payload)

**Problem**: In `project_archiver.py`, the function `_archive_fits()` (around line 290)
stores only the inner `fit_state_payload` dict. The fit group envelope is lost:
- `id` (fit UID)
- `name` (user-visible name)
- `model_name` (which model class, e.g. "FRET: FD (Gaussian)")
- `plot_state` (axis ranges, log scale, visible curves)
- `fit_range` (data range for chi-squared)
- `local_fits` list structure

**Fix**: When creating the `fit_result` artifact, include the full fit record in
`metadata_json`, not just the payload.

**File**: `chisurf/core/mfdb/project_archiver.py`

**What to change**: Find where `fit_state_payload` is stored as artifact data. Change it
to store the complete fit record dict (as returned by `make_fit_record()` in
`chisurf/core/project/fit_state.py`). The artifact's `metadata_json` must include:

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

The artifact data (stored in object store) should be the full `fit_record` dict as JSON.

### Task 2: Store Project-Level Metadata

**Problem**: In `archive_project_to_mfdb()`, the metadata dict (around line 162) only has
`project_id`, `version_number`, `fit_count`, `dataset_count`. Missing:
- `chisurf_version`
- `project_format_version`
- `description`
- `created` timestamp
- `ui_state`
- `experiments`

**Fix**: Add these fields to the metadata dict.

**File**: `chisurf/core/mfdb/project_archiver.py`

**What to change**: Find the `metadata={}` dict in `archive_project_to_mfdb()`. Add:

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

### Task 3: Fix Dataset Restore

**Problem 1**: In `restore_project_from_artifacts()` (around line 580), the code filters
by `kind == "processed_data" and role == "dataset"`. But `role` comes from the
`mfdb_operation_artifact` junction table, not from the artifact itself. If the query
doesn't join properly, `role` is always empty and ALL datasets are silently dropped.

**Problem 2**: The fallback `ds_id = ds_id.get("ds_id", role)` means all datasets get
key `"dataset"`, so only the last one survives.

**Fix**:

**File**: `chisurf/core/mfdb/project_archiver.py`

**Step 1**: Check how `db.get_operation_artifacts()` works. Open
`chisurf/core/mfdb/repository.py` and find this method. Verify it returns rows with a
`role` column from `mfdb_operation_artifact`. If it doesn't, fix the query to JOIN and
include `role`.

**Step 2**: In `restore_project_from_artifacts()`, change the dataset key logic:

```python
# OLD (broken):
ds_id = (art.get("metadata_json") or {})
if isinstance(ds_id, str):
    ds_id = _json_loads(ds_id) or {}
ds_id = ds_id.get("ds_id", role)

# NEW (fixed):
meta = art.get("metadata_json") or {}
if isinstance(meta, str):
    meta = _json_loads(meta) or {}
ds_id = meta.get("ds_id") or meta.get("dataset_uid") or art.get("artifact_id") or str(uuid.uuid4())
```

**Step 3**: Also store dataset metadata during archiving. In `_archive_datasets()`, make
sure each dataset artifact has `metadata_json` containing:

```python
{
    "ds_id": dataset_uid,
    "filename": dataset.get("filename", ""),
    "reader_class": dataset.get("reader_class", ""),
}
```

### Task 4: Restore Fit Records with Full Structure

**Problem**: `restore_project_from_artifacts()` returns a flat list of fit_state dicts.
The project loader expects fit group records with `id`, `name`, `model_name`, `local_fits`.

**Fix**: Change the restore function to read fit_result artifacts as full fit records
(since Task 1 now stores them that way).

**File**: `chisurf/core/mfdb/project_archiver.py`

**What to change**: In the fit restoration loop (inside `restore_project_from_artifacts()`),
instead of appending the raw artifact data as a flat dict, reconstruct the fit record:

```python
# Read the artifact data from object store
fit_data = _json_loads(artifact_data_bytes)

# If it's a full fit record (has "id" and "local_fits"), use as-is
if isinstance(fit_data, dict) and "local_fits" in fit_data:
    fits.append(fit_data)
else:
    # Legacy fallback: wrap in a fit record structure
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

### Task 5: Use Chinet Sessions on Restore

**Problem**: `restore_project_handler()` in
`chisurf/plugins/core/project_browser/backend/services.py` (around line 387) discards
the `chinet_sessions` returned by `restore_project_from_artifacts()`.

**Fix**: Pass chinet sessions into the restored project payload so the project loader
can re-establish parameter dependencies.

**File**: `chisurf/plugins/core/project_browser/backend/services.py`

**What to change**: Find where the restored payload is constructed (around line 375-390).
Add the chinet sessions:

```python
# OLD:
payload = {
    "datasets": restored["datasets"],
    "fits": restored["fits"],
    "experiments": {},
}

# NEW:
payload = {
    "datasets": restored["datasets"],
    "fits": restored["fits"],
    "experiments": restored.get("experiments", {}),
    "ui_state": restored.get("ui_state", {}),
    "chinet_sessions": restored.get("chinet_sessions", []),
}
```

### Task 6: Query Parameters and Edges on Restore

**Problem**: `restore_project_from_artifacts()` never reads `mfdb_parameter` or
`mfdb_edge` tables. Parameters are stored but lost on restore.

**Fix**: After collecting fits, query for parameters and dependency edges.

**File**: `chisurf/core/mfdb/project_archiver.py`

**What to add**: At the end of `restore_project_from_artifacts()`, before the return:

```python
# Query parameters for each fit operation
all_parameters = {}
for fit_op_id in fit_operation_ids:
    rows = db.con.execute(
        "SELECT * FROM mfdb_parameter WHERE operation_id = ?",
        (fit_op_id,)
    ).fetchall()
    all_parameters[fit_op_id] = [dict(r) for r in rows]

# Query dependency edges
dependency_edges = []
rows = db.con.execute(
    "SELECT * FROM mfdb_edge WHERE relationship_type = 'parameter_depends_on'"
).fetchall()
dependency_edges = [dict(r) for r in rows]

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

Note: You need to collect `fit_operation_ids` as you iterate fits. Add a list at the top
of the function and append each fit operation's ID as you process it.

Also collect `project_metadata` from the project operation's metadata_json field.

### Task 7: Write Round-Trip Test

**File to create**: `test/fio/test_mfdb_project_roundtrip.py`

```python
"""Test that archive -> restore produces identical project data."""
import json
import os
import tempfile
import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.project_archiver import (
    archive_project_to_mfdb,
    restore_project_from_artifacts,
)


@pytest.fixture
def db():
    """Create a temporary MFDB database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        obj_root = os.path.join(tmpdir, "objects")
        os.makedirs(obj_root)
        database = MFDatabase(db_path, object_store_root=obj_root)
        yield database
        database.close()


def _make_test_payload():
    """Create a minimal project payload with 2 datasets and 2 fits."""
    return {
        "chisurf_version": "25.1.0",
        "project_format_version": 4,
        "description": "Test project",
        "created": "2025-01-01T00:00:00",
        "datasets": {
            "ds_001": {
                "uid": "ds_001",
                "filename": "sample.ptu",
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [100.0, 80.0, 60.0, 40.0],
            },
            "ds_002": {
                "uid": "ds_002",
                "filename": "donor_only.ptu",
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [200.0, 150.0, 100.0, 50.0],
            },
        },
        "fits": [
            {
                "id": "fit_001",
                "name": "DA sample fit",
                "model_name": "FRET: FD (Gaussian)",
                "local_fits": [
                    {
                        "parameters": {
                            "p1": {"value": 5.0, "fixed": False, "bounds": [0, 10]},
                            "p2": {"value": 3.0, "fixed": True, "bounds": [0, 5]},
                        }
                    }
                ],
            },
            {
                "id": "fit_002",
                "name": "Donor only fit",
                "model_name": "Lifetime",
                "local_fits": [
                    {
                        "parameters": {
                            "tau1": {"value": 4.0, "fixed": False, "bounds": [0, 20]},
                        }
                    }
                ],
            },
        ],
        "ui_state": {"current_fit_index": 0},
        "experiments": {},
    }


def test_roundtrip_preserves_datasets(db):
    payload = _make_test_payload()
    version_id = archive_project_to_mfdb(
        db=db,
        payload=payload,
        project_name="test_project",
    )
    restored = restore_project_from_artifacts(db, version_id)

    assert len(restored["datasets"]) == 2
    for ds_id, ds_data in payload["datasets"].items():
        assert ds_id in restored["datasets"], f"Dataset {ds_id} missing after restore"


def test_roundtrip_preserves_fit_structure(db):
    payload = _make_test_payload()
    version_id = archive_project_to_mfdb(
        db=db,
        payload=payload,
        project_name="test_project",
    )
    restored = restore_project_from_artifacts(db, version_id)

    assert len(restored["fits"]) == 2
    for i, fit in enumerate(restored["fits"]):
        assert "id" in fit, f"Fit {i} missing 'id'"
        assert "name" in fit, f"Fit {i} missing 'name'"
        assert "model_name" in fit, f"Fit {i} missing 'model_name'"
        assert "local_fits" in fit, f"Fit {i} missing 'local_fits'"


def test_roundtrip_preserves_metadata(db):
    payload = _make_test_payload()
    version_id = archive_project_to_mfdb(
        db=db,
        payload=payload,
        project_name="test_project",
    )
    restored = restore_project_from_artifacts(db, version_id)

    assert restored.get("ui_state") == payload["ui_state"]


def test_roundtrip_two_datasets_not_collapsed(db):
    """Regression: previously all datasets collapsed to key 'dataset'."""
    payload = _make_test_payload()
    version_id = archive_project_to_mfdb(
        db=db,
        payload=payload,
        project_name="test_project",
    )
    restored = restore_project_from_artifacts(db, version_id)

    dataset_keys = list(restored["datasets"].keys())
    assert len(set(dataset_keys)) == 2, f"Dataset keys not unique: {dataset_keys}"
```

Adapt the test to match the actual function signatures. The key assertions are:
1. Two datasets in -> two datasets out (not collapsed)
2. Fit records have id, name, model_name, local_fits
3. UI state survives the round-trip
4. Dataset UIDs are preserved

## Definition of Done

- [ ] `archive_project_to_mfdb()` stores full fit records with id/name/model_name
- [ ] `archive_project_to_mfdb()` stores project-level metadata (version, ui_state, experiments)
- [ ] `restore_project_from_artifacts()` returns fit group records, not flat dicts
- [ ] `restore_project_from_artifacts()` returns all datasets with unique keys
- [ ] `restore_project_from_artifacts()` returns parameters and dependency edges
- [ ] Restore handler in project_browser uses chinet sessions
- [ ] Round-trip test passes
