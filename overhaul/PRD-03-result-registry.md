# PRD-03: Result Registry

## Goal

Create a single function that any plugin can call to register its output in MFDB. Plugins
should not need to understand MFDB internals -- just call `register_result()` and the
data is archived with full provenance.

## Background

Read these files before starting:
- `chisurf/core/mfdb/repository.py` -- `register_artifact()`, `record_operation()`, `put_object()`
- `chisurf/core/mfdb/models.py` -- vocabulary constants (`ARTIFACT_KINDS`, `OPERATION_TYPES`, etc.)
- `chisurf/core/mfdb/object_store.py` -- content-addressed storage

79 of 87 plugins produce data but don't write to MFDB. We need a dead-simple API.

## Tasks

### Task 1: Create the Result Registry Module

**File to create**: `chisurf/core/mfdb/result_registry.py`

```python
"""Simple API for plugins to register results in MFDB.

Usage:
    from chisurf.core.mfdb.result_registry import register_result

    result_id = register_result(
        kind="burst_data",
        data=my_dataframe,              # or file path or bytes
        sample_id="my_sample",
        parent_artifact_id="input_123",
        operation_type="burst_selection",
        parameters={"min_photons": 50},
        metadata={"algorithm": "APBS"},
    )
"""
import json
import uuid
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def register_result(
    kind: str,
    data: Union[str, bytes, Path, "pandas.DataFrame", dict, None] = None,
    sample_id: str = "",
    parent_artifact_id: str = "",
    operation_type: str = "",
    parameters: Optional[dict] = None,
    metadata: Optional[dict] = None,
    data_format: str = "",
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a result in MFDB. Returns the artifact_id.

    Args:
        kind: What type of result. Must be a valid artifact_kind from models.py.
            Common values: "raw_measurement", "processed_data", "burst_data",
            "fit_result", "calibration_data", "spectrum_data", "irf_data",
            "background_data", "correlation_data".
        data: The result data. Can be:
            - str or Path: path to a file (will be copied to object store)
            - bytes: raw bytes (stored in object store)
            - dict: serialized as JSON bytes and stored
            - pandas.DataFrame: serialized as JSON bytes and stored
            - None: no data stored (metadata-only artifact)
        sample_id: Which sample this result belongs to. Creates a
            "measured_sample" edge. Strongly encouraged but not required.
        parent_artifact_id: The input artifact that produced this result.
            Creates a "derived_from" edge.
        operation_type: What kind of processing produced this result.
            Common values: "measurement", "burst_selection", "local_fit",
            "calibration", "correlation", "histogram_construction".
            If empty, defaults to "processing".
        parameters: Dict of parameter name -> value pairs to record in
            mfdb_parameter table. Values can be numbers or dicts with
            keys: value, error, fixed, bounds.
        metadata: Additional metadata stored in the artifact's metadata_json.
        data_format: File format hint (e.g. "ptu", "json", "csv", "npy").
            Auto-detected from file extension if data is a path.
        db: MFDatabase instance. If None, uses the global database from
            the database connector.

    Returns:
        artifact_id (str): UUID of the created artifact.
    """
    if db is None:
        db = _get_global_db()
    if db is None:
        logger.warning("No MFDB database available. Result not registered.")
        return ""

    artifact_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    object_uuid = None
    storage_mode = "inline_json"

    # Store data in object store
    if data is not None:
        object_uuid, storage_mode, data_format = _store_data(
            db, data, data_format
        )

    # Create artifact record
    db.register_artifact(
        artifact_id=artifact_id,
        artifact_kind=kind,
        data_format=data_format,
        storage_mode=storage_mode,
        object_uuid=object_uuid,
        metadata_json=json.dumps(metadata) if metadata else None,
    )

    # Create operation record
    op_type = operation_type or "processing"
    db.record_operation(
        operation_id=operation_id,
        operation_type=op_type,
        status="succeeded",
        metadata=metadata or {},
    )

    # Link operation -> artifact (output)
    db.record_operation_link(
        operation_id=operation_id,
        artifact_id=artifact_id,
        direction="output",
        role=kind,
    )

    # Link parent artifact -> operation (input)
    if parent_artifact_id:
        db.record_operation_link(
            operation_id=operation_id,
            artifact_id=parent_artifact_id,
            direction="input",
            role="source",
        )
        # Also create a derived_from edge
        db.add_edge(
            source_id=artifact_id,
            source_type="artifact",
            target_id=parent_artifact_id,
            target_type="artifact",
            relationship_type="derived_from",
        )

    # Link to sample
    if sample_id:
        from chisurf.core.mfdb.sample_manager import link_artifact_to_sample
        link_artifact_to_sample(db, artifact_id, sample_id)

    # Record parameters
    if parameters:
        _record_parameters(db, operation_id, parameters)

    db.con.commit()
    logger.info(
        "Registered result: kind=%s, artifact_id=%s, operation=%s",
        kind, artifact_id, op_type,
    )
    return artifact_id


def _store_data(db, data, data_format):
    """Store data in the object store. Returns (object_uuid, storage_mode, data_format)."""
    import pandas as pd

    if isinstance(data, (str, Path)):
        path = Path(data)
        if not data_format:
            data_format = path.suffix.lstrip(".")
        ref = db.object_store.put_from_path(str(path))
        return ref.uuid, "object_store", data_format

    if isinstance(data, bytes):
        ref = db.object_store.put_bytes(data)
        return ref.uuid, "object_store", data_format or "bin"

    if isinstance(data, pd.DataFrame):
        blob = data.to_json(orient="records").encode("utf-8")
        ref = db.object_store.put_bytes(blob)
        return ref.uuid, "object_store", "json"

    if isinstance(data, dict):
        blob = json.dumps(data).encode("utf-8")
        ref = db.object_store.put_bytes(blob)
        return ref.uuid, "object_store", "json"

    raise TypeError(f"Unsupported data type: {type(data)}")


def _record_parameters(db, operation_id, parameters):
    """Write parameters to mfdb_parameter table."""
    for name, value in parameters.items():
        if isinstance(value, dict):
            db.record_parameter(
                operation_id=operation_id,
                parameter_name=name,
                parameter_value=value.get("value"),
                standard_error=value.get("error"),
                parameter_type="fixed" if value.get("fixed") else "free",
                lower_bound=value.get("bounds", [None, None])[0] if value.get("bounds") else None,
                upper_bound=value.get("bounds", [None, None])[1] if value.get("bounds") else None,
            )
        else:
            db.record_parameter(
                operation_id=operation_id,
                parameter_name=name,
                parameter_value=float(value) if isinstance(value, (int, float)) else None,
                parameter_type="fixed",
            )


def _get_global_db():
    """Get the global MFDatabase instance from the database connector.

    Returns None if no database is available (headless mode, tests, etc.)
    """
    try:
        from chisurf.core.mfdb.database_resolver import resolve_database_path
        from chisurf.core.mfdb.repository import MFDatabase
        # Try to get the already-open instance
        # This depends on how the database connector plugin works.
        # Check chisurf/plugins/core/database_connector/ for the global instance.
        # For now, return None and let callers pass db explicitly.
        return None
    except Exception:
        return None
```

**Important**: The function signatures for `register_artifact()`, `record_operation()`,
etc. must match what's in `repository.py`. Read that file and adapt the calls. The names
above are best guesses -- verify them.

### Task 2: Implement _get_global_db()

The result registry needs access to the global database instance. Look at how
`chisurf/plugins/core/database_connector/` manages the database connection.

**File**: `chisurf/core/mfdb/result_registry.py` (update `_get_global_db()`)

Options:
1. If there's a global/singleton database instance, import and return it
2. If the database connector uses a service registry, query it
3. If neither exists, create a simple module-level variable:

```python
_GLOBAL_DB: Optional["MFDatabase"] = None

def set_global_db(db: "MFDatabase"):
    """Called by the database connector when the DB is opened."""
    global _GLOBAL_DB
    _GLOBAL_DB = db

def _get_global_db():
    return _GLOBAL_DB
```

Then modify the database connector plugin to call `set_global_db()` when it opens the
database.

### Task 3: Add Missing Artifact Kinds and Operation Types

**File**: `chisurf/core/mfdb/models.py`

Check the `ARTIFACT_KINDS` list and add these if missing:
- `"correlation_data"` -- FCS correlation functions
- `"background_data"` -- buffer/background measurements
- `"irf_data"` -- instrument response functions
- `"spectrum_data"` -- absorption/emission spectra
- `"burst_data"` -- burst selection results
- `"image_data"` -- CLSM images
- `"trace_data"` -- intensity time traces

Check the `OPERATION_TYPES` list and add these if missing:
- `"burst_selection"`
- `"histogram_construction"`
- `"background_correction"`
- `"correlation"`
- `"image_analysis"`
- `"population_selection"`
- `"spectrum_measurement"`
- `"processing"` -- generic fallback

Also check `schema.py` for any CHECK constraints on these columns. If the constraints
list specific allowed values, add the new values there too.

### Task 4: Add Convenience Wrappers for Common Results

Add these to `result_registry.py` after the main `register_result()` function:

```python
def register_raw_measurement(
    file_path: str,
    sample_id: str = "",
    metadata: Optional[dict] = None,
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a raw measurement file (TTTR, SPC, etc.)."""
    return register_result(
        kind="raw_measurement",
        data=file_path,
        sample_id=sample_id,
        operation_type="measurement",
        metadata=metadata,
        db=db,
    )


def register_processed_data(
    data,
    parent_artifact_id: str,
    sample_id: str = "",
    operation_type: str = "processing",
    parameters: Optional[dict] = None,
    metadata: Optional[dict] = None,
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register processed data derived from another artifact."""
    return register_result(
        kind="processed_data",
        data=data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type=operation_type,
        parameters=parameters,
        metadata=metadata,
        db=db,
    )


def register_fit_result(
    fit_data: dict,
    parent_artifact_id: str,
    sample_id: str = "",
    parameters: Optional[dict] = None,
    metadata: Optional[dict] = None,
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a fit result."""
    return register_result(
        kind="fit_result",
        data=fit_data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type="local_fit",
        parameters=parameters,
        metadata=metadata,
        db=db,
    )


def register_calibration(
    data,
    calibration_type: str,
    sample_id: str = "",
    parent_artifact_id: str = "",
    parameters: Optional[dict] = None,
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a calibration result (g-factor, gamma, crosstalk, etc.)."""
    return register_result(
        kind="calibration_data",
        data=data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type="calibration",
        parameters=parameters,
        metadata={"calibration_type": calibration_type},
        db=db,
    )
```

### Task 5: Write Tests

**File to create**: `test/fio/test_result_registry.py`

```python
"""Tests for result_registry.py."""
import json
import os
import tempfile
import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import (
    register_result,
    register_raw_measurement,
    register_processed_data,
    register_fit_result,
    register_calibration,
)


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        obj_root = os.path.join(tmpdir, "objects")
        os.makedirs(obj_root)
        database = MFDatabase(db_path, object_store_root=obj_root)
        yield database
        database.close()


def test_register_result_with_dict_data(db):
    art_id = register_result(
        kind="processed_data",
        data={"x": [1, 2, 3], "y": [4, 5, 6]},
        operation_type="processing",
        db=db,
    )
    assert art_id
    # Verify artifact exists in DB
    row = db.con.execute(
        "SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,)
    ).fetchone()
    assert row is not None
    assert row[0] == "processed_data"


def test_register_result_with_file(db, tmp_path):
    # Create a dummy file
    f = tmp_path / "test.ptu"
    f.write_bytes(b"fake ptu data")

    art_id = register_result(
        kind="raw_measurement",
        data=str(f),
        operation_type="measurement",
        db=db,
    )
    assert art_id
    row = db.con.execute(
        "SELECT data_format FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,)
    ).fetchone()
    assert row[0] == "ptu"


def test_register_result_creates_provenance_edge(db):
    parent_id = register_result(
        kind="raw_measurement",
        data=b"raw data",
        operation_type="measurement",
        db=db,
    )
    child_id = register_result(
        kind="processed_data",
        data=b"processed",
        parent_artifact_id=parent_id,
        operation_type="processing",
        db=db,
    )
    # Check derived_from edge exists
    row = db.con.execute(
        """SELECT relationship_type FROM mfdb_edge
           WHERE source_id = ? AND target_id = ?""",
        (child_id, parent_id)
    ).fetchone()
    assert row is not None
    assert row[0] == "derived_from"


def test_register_result_with_sample(db):
    from chisurf.core.mfdb.sample_manager import create_sample, get_artifacts_for_sample
    from chisurf.core.mfdb.models import SampleDefinition

    sample_id = create_sample(db, SampleDefinition(name="test_sample"))

    art_id = register_result(
        kind="raw_measurement",
        data=b"data",
        sample_id=sample_id,
        operation_type="measurement",
        db=db,
    )

    arts = get_artifacts_for_sample(db, sample_id)
    assert art_id in arts


def test_register_result_with_parameters(db):
    art_id = register_result(
        kind="fit_result",
        data={"chi2": 1.05},
        operation_type="local_fit",
        parameters={
            "tau1": {"value": 4.0, "error": 0.1, "fixed": False, "bounds": [0, 20]},
            "amplitude": 0.85,
        },
        db=db,
    )

    # Check parameters exist
    rows = db.con.execute(
        """SELECT parameter_name, parameter_value FROM mfdb_parameter
           WHERE operation_id IN (
               SELECT operation_id FROM mfdb_operation_artifact
               WHERE artifact_id = ? AND direction = 'output'
           )""",
        (art_id,)
    ).fetchall()
    names = {r[0] for r in rows}
    assert "tau1" in names
    assert "amplitude" in names


def test_register_result_no_db_returns_empty():
    """When no DB is available, return empty string without error."""
    art_id = register_result(
        kind="processed_data",
        data=b"data",
        db=None,
    )
    # Should return "" since _get_global_db returns None
    # and db parameter is None
    assert art_id == ""
```

**Important**: Adapt all SQL queries and method calls to match the actual repository API.
Read `repository.py` to verify method names and signatures.

## Definition of Done

- [ ] `result_registry.py` exists with `register_result()` and convenience wrappers
- [ ] `_get_global_db()` returns the active database instance
- [ ] All missing artifact kinds and operation types are added to `models.py`
- [ ] Provenance edges are created correctly (derived_from, measured_sample)
- [ ] Parameters are stored in mfdb_parameter
- [ ] All tests pass
