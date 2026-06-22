# PRD-03: Result Registry

> ⛔ **Blocked by PRD-030 (Result Payload Formats & Codecs).**
> PRD-030 must be completed first. The result registry stores and retrieves
> scientific payloads (FCS curves, spectra, TTTR, TCSPC decays, burst tables,
> …), but those payload formats are not yet formally defined. Until PRD-030
> establishes the typed, msgpack-based payload schemas and codec registry,
> `register_result()` can only serialize data ad-hoc (lossy JSON), which is the
> exact weakness recorded in `overhaul/CODE_REPORT.md`. Do **not** start PRD-03
> until PRD-030 lands; PRD-03's `_store_data()` and any `read_result()` must use
> the PRD-030 codecs rather than `to_json()`/`json.dumps()`.

## Goal

Create a single function that any plugin can call to register its output in MFDB. Plugins
should not need to understand MFDB internals -- just call `register_result()` and the
data is archived with full provenance (object stored, artifact + operation rows created,
input/output and `derived_from` / `measured_sample` edges wired, parameters recorded).

79 of 87 plugins produce data but don't write to MFDB. This PRD gives them a dead-simple
API and a single, well-tested choke point so provenance is uniform.

## Background

Read these files before starting:
- `overhaul/PRD-030-result-payload-formats.md` -- **prerequisite.** Defines the typed,
  msgpack-based payload schemas and the codec registry that `register_result()` must use
  for (de)serialization. PRD-03 cannot meet its DoD without it.
- `overhaul/CODE_REPORT.md` -- records why PRD-030 must precede this work.
- `chisurf/core/mfdb/repository.py` -- the `MFDatabase` class. **This PRD has been written
  against the real method signatures below.** Verify them before coding.
- `chisurf/core/mfdb/models.py` -- vocabulary tuples (`ARTIFACT_KINDS`, `OPERATION_TYPES`,
  `RELATIONSHIP_TYPES`, `STATUS_VALUES`, `DIRECTIONS`, `PARAMETER_TYPES`).
- `chisurf/core/mfdb/object_store.py` -- content-addressed storage (wrapped by
  `MFDatabase.put_object`).
- `chisurf/core/mfdb/sample_manager.py` -- `link_artifact_to_sample`,
  `get_artifacts_for_sample`, `create_sample`, `SampleDefinition`.
- `chisurf/core/mfdb/database_resolver.py` -- `resolve_database_path`, `object_store_root`.
- `chisurf/plugins/core/database_connector/services.py` -- the `DatabaseConnector`
  singleton (`_connector`) that owns the active DB connection.
- `chisurf/core/mfdb/project_archiver.py` -- existing archiver. `register_result()` must
  **not** duplicate its responsibilities; it is the lightweight per-plugin path, the
  archiver is the whole-project path. Both call the same repository primitives.

## Verified Repository API (against repository.py)

> ⚠️ The original draft of this PRD used invented method names. These are the **actual**
> signatures. Use exactly these.

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

Relevant vocabulary (from `models.py`, current values):
- `ARTIFACT_KINDS` includes: `raw_measurement`, `processed_data`, `analysis_result`,
  `fit_result`, `fcs_correlation`, `irf_curve`, `spectra`, `burst_table`, `tcspc_decay`,
  `pda_histogram`, `anisotropy_curve`, `selection_mask`, `visualization`,
  `external_reference`, plus many legacy aliases. **`artifact_kind` is extensible** --
  unknown values are accepted and auto-registered by `validate_extensible_vocab`.
- `OPERATION_TYPES` includes: `measurement_import`, `burst_selection`, `fcs_correlation`,
  `tcspc_fitting`, `model_fitting`, `tcspc_histogram_computation`,
  `pda_histogram_computation`, `pch_histogram_computation`, `local_fit`, `global_fit`,
  `analysis`, plus legacy. **`operation_type` is also extensible.**
- `STORAGE_MODES`: `local_file`, `embedded_json`, `embedded_blob`, `url`, ...
- `RELATIONSHIP_TYPES`: `derived_from`, `measured_sample`, `supersedes`, `linked_to`, ...
- `STATUS_VALUES`: `pending`, `running`, `succeeded`, `failed`, ...

## Tasks

### Task 1: Create the Result Registry Module

**File to create**: `chisurf/core/mfdb/result_registry.py`

```python
"""Simple API for plugins to register results in MFDB.

This is the lightweight per-plugin path. It wraps the canonical MFDatabase
primitives (put_object / register_artifact / record_operation /
record_operation_link / record_parameter / add_edge) into one call so plugins
do not need to know MFDB internals.

Usage:
    from chisurf.core.mfdb.result_registry import register_result

    artifact_id = register_result(
        kind="burst_table",
        data=my_dataframe,              # or file path / bytes / dict
        sample_id="my_sample",
        parent_artifact_id="input_123",
        operation_type="burst_selection",
        parameters={"min_photons": 50},
        metadata={"algorithm": "APBS"},
    )

Design rules:
- If no database is available (headless, tests without a DB), return "" and log
  a warning. Never raise into a plugin's normal flow.
- Provenance edges: parent_artifact_id -> derived_from; sample_id -> measured_sample.
- All writes go through repository methods that self-commit; do not commit here.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import pandas
    from chisurf.core.mfdb.repository import MFDatabase

logger = logging.getLogger(__name__)


def register_result(
    kind: str,
    data: Union[str, bytes, "Path", "pandas.DataFrame", dict, None] = None,
    sample_id: str = "",
    parent_artifact_id: str = "",
    operation_type: str = "",
    parameters: Optional[dict] = None,
    metadata: Optional[dict] = None,
    data_format: str = "",
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a result in MFDB. Returns the artifact_id (or "" if no DB).

    Parameters
    ----------
    kind : str
        Artifact kind. Prefer a value from ``models.ARTIFACT_KINDS`` (e.g.
        ``"raw_measurement"``, ``"processed_data"``, ``"fit_result"``,
        ``"fcs_correlation"``, ``"burst_table"``, ``"spectra"``, ``"irf_curve"``,
        ``"tcspc_decay"``). Unknown kinds are accepted (the vocabulary is
        extensible) but reuse an existing name where one fits.
    data : str | Path | bytes | dict | pandas.DataFrame | None
        The payload. ``str``/``Path`` is treated as a file and copied into the
        object store; ``bytes`` is stored as-is; ``dict``/``DataFrame`` is
        serialized to JSON bytes. ``None`` creates a metadata-only artifact.
    sample_id : str
        If set, a ``measured_sample`` edge is created (artifact -> sample).
    parent_artifact_id : str
        If set, the operation gets an ``input`` link to this artifact and a
        ``derived_from`` edge (new artifact -> parent) is created.
    operation_type : str
        Operation vocabulary value. Defaults to ``"analysis"`` if empty.
    parameters : dict, optional
        ``name -> value`` or ``name -> {value, error, fixed, bounds, units}``.
        Recorded in ``mfdb_parameter``.
    metadata : dict, optional
        Stored on both the artifact and the operation as JSON.
    data_format : str, optional
        Format hint (``"ptu"``, ``"json"``, ``"csv"``, ...). Auto-detected from
        a file suffix when ``data`` is a path.
    db : MFDatabase, optional
        Explicit DB. If ``None``, uses the active global DB (see ``_get_global_db``).

    Returns
    -------
    str
        The created ``artifact_id``, or ``""`` if no database is available.
    """
    if db is None:
        db = _get_global_db()
    if db is None:
        logger.warning("register_result: no MFDB available; result not registered (kind=%s)", kind)
        return ""

    artifact_id = str(uuid.uuid4())
    operation_id = str(uuid.uuid4())
    object_uuid: Optional[str] = None
    storage_mode = "embedded_json"
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    mime_type: Optional[str] = None

    try:
        # 1. Store the payload (if any) in the object store.
        if data is not None:
            object_uuid, storage_mode, data_format, size_bytes, checksum, mime_type = (
                _store_data(db, data, data_format)
            )

        # 2. Create the artifact row.
        db.register_artifact(
            artifact_id=artifact_id,
            artifact_kind=kind,
            data_format=data_format or None,
            storage_mode=storage_mode,
            object_uuid=object_uuid,
            size_bytes=size_bytes,
            checksum=checksum,
            mime_type=mime_type,
            metadata=metadata or None,
        )

        # 3. Create the operation row.
        op_type = operation_type or "analysis"
        db.record_operation(
            operation_id=operation_id,
            operation_type=op_type,
            status="succeeded",
            metadata=metadata or None,
        )

        # 4. Link operation -> artifact (output).
        db.record_operation_link(
            operation_id=operation_id,
            artifact_id=artifact_id,
            direction="output",
            role=kind,
        )

        # 5. Wire the parent artifact (input link + derived_from edge).
        if parent_artifact_id:
            db.record_operation_link(
                operation_id=operation_id,
                artifact_id=parent_artifact_id,
                direction="input",
                role="source",
            )
            db.add_edge(
                source_node_type="artifact",
                source_node_id=artifact_id,
                target_node_type="artifact",
                target_node_id=parent_artifact_id,
                relationship_type="derived_from",
            )

        # 6. Link to a sample (measured_sample edge).
        if sample_id:
            from chisurf.core.mfdb.sample_manager import link_artifact_to_sample
            link_artifact_to_sample(db, artifact_id, sample_id)

        # 7. Record parameters.
        if parameters:
            _record_parameters(db, operation_id, parameters)

    except Exception as exc:  # never raise into a plugin's normal flow
        logger.warning("register_result failed (kind=%s): %s", kind, exc, exc_info=True)
        return ""

    logger.info(
        "Registered result: kind=%s artifact=%s operation=%s", kind, artifact_id, op_type
    )
    return artifact_id


def _store_data(db, data, data_format):
    """Store data in the object store.

    Returns (object_uuid, storage_mode, data_format, size_bytes, checksum, mime_type).

    NOTE (PRD-030): the JSON/`to_json` serialization shown below is a placeholder.
    Once PRD-030 lands, replace the per-type branches with a call into the codec
    registry: ``blob, fmt = encode_payload(kind, data)`` (msgpack-based), so the
    payload round-trips losslessly with a well-defined schema.
    """
    # Local import so the module imports cleanly without pandas installed.
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None

    if isinstance(data, (str, Path)):
        path = Path(data)
        fmt = data_format or path.suffix.lstrip(".")
        ref = db.put_object(path=path, filename=path.name)
        return ref["object_uuid"], "local_file", fmt, ref["size_bytes"], ref["content_md5"], None

    if isinstance(data, bytes):
        ref = db.put_object(data=data, filename="payload.bin")
        return ref["object_uuid"], "embedded_blob", data_format or "bin", ref["size_bytes"], ref["content_md5"], None

    if pd is not None and isinstance(data, pd.DataFrame):
        blob = data.to_json(orient="records").encode("utf-8")
        ref = db.put_object(data=blob, filename="dataframe.json", mime_type="application/json")
        return ref["object_uuid"], "embedded_json", "json", ref["size_bytes"], ref["content_md5"], "application/json"

    if isinstance(data, dict):
        blob = json.dumps(data, default=str).encode("utf-8")
        ref = db.put_object(data=blob, filename="data.json", mime_type="application/json")
        return ref["object_uuid"], "embedded_json", "json", ref["size_bytes"], ref["content_md5"], "application/json"

    raise TypeError(f"Unsupported data type for register_result: {type(data)!r}")


def _record_parameters(db, operation_id, parameters):
    """Write parameters to mfdb_parameter via record_parameter()."""
    for name, value in parameters.items():
        param_uuid = str(uuid.uuid4())
        if isinstance(value, dict):
            bounds = value.get("bounds") or [None, None]
            db.record_parameter(
                parameter_uuid=param_uuid,
                operation_id=operation_id,
                name=name,
                value=value.get("value"),
                standard_error=value.get("error"),
                lower_bound=bounds[0],
                upper_bound=bounds[1],
                units=value.get("units"),
                parameter_type="fixed" if value.get("fixed") else "free",
            )
        else:
            numeric = float(value) if isinstance(value, (int, float)) else None
            db.record_parameter(
                parameter_uuid=param_uuid,
                operation_id=operation_id,
                name=name,
                value=numeric,
                parameter_type="fixed",
            )
```

### Task 2: Implement `_get_global_db()`

The registry must reach the active database without each plugin passing one in. There is
already a module-level `DatabaseConnector` singleton (`_connector`) in
`chisurf/plugins/core/database_connector/services.py`, but its `_db` is only populated
after `.open()` is called. Provide a layered resolver plus an explicit override hook.

**File**: `chisurf/core/mfdb/result_registry.py` (append)

```python
_GLOBAL_DB: Optional["MFDatabase"] = None


def set_global_db(db: Optional["MFDatabase"]) -> None:
    """Explicitly set (or clear) the global DB. Useful for tests and headless runs."""
    global _GLOBAL_DB
    _GLOBAL_DB = db


def _get_global_db() -> Optional["MFDatabase"]:
    """Resolve the active MFDatabase.

    Resolution order:
    1. An explicitly set global DB (``set_global_db``).
    2. The database connector's open connection, if any.
    3. A fresh connection to the resolved user database path.

    Returns None only if even opening the user DB fails (truly headless).
    """
    if _GLOBAL_DB is not None:
        return _GLOBAL_DB

    # 2. Reuse the database connector's open handle when available.
    try:
        from chisurf.plugins.core.database_connector.services import _connector
        if getattr(_connector, "_db", None) is not None:
            return _connector._db
    except Exception:
        pass

    # 3. Fall back to opening the resolved user database.
    try:
        from chisurf.core.mfdb.database_resolver import resolve_database_path
        from chisurf.core.mfdb.repository import MFDatabase
        return MFDatabase(resolve_database_path())
    except Exception as exc:
        logger.warning("result_registry: could not open MFDB: %s", exc)
        return None
```

**Note on connection lifetime**: option 3 opens a fresh `MFDatabase` per call. Because
`MFDatabase` is a thin SQLite wrapper and all writes auto-commit, this is safe but not
free. For hot paths (a plugin registering many results in a loop), the plugin should open
one `MFDatabase` and pass it via the `db=` argument, or call `set_global_db()` once.
Document this in the function docstring.

### Task 3: Reconcile Artifact Kinds and Operation Types (do NOT blindly add)

The original draft proposed adding kinds like `correlation_data`, `irf_data`,
`spectrum_data`, `burst_data`. **Most already exist under different names.** Reuse the
canonical name; only add genuinely new vocabulary.

| Proposed (draft) | Use existing canonical | Action |
|------------------|------------------------|--------|
| `correlation_data` | `fcs_correlation` | reuse |
| `irf_data` | `irf_curve` | reuse |
| `spectrum_data` | `spectra` | reuse |
| `burst_data` | `burst_table` | reuse |
| `trace_data` | -- | add `trace_data` |
| `image_data` | -- | add `image_data` |
| `background_data` | -- | add `background_data` |

For operation types, reuse `burst_selection`, `fcs_correlation`,
`tcspc_histogram_computation`, etc. Add only if missing:
- `histogram_construction` (or reuse `tcspc_histogram_computation` where specific)
- `background_correction`
- `image_analysis`
- `population_selection`

Because both vocabularies are **extensible** (`validate_extensible_vocab`), you have two
ways to introduce a genuinely new value:
1. Add it to the tuple in `chisurf/core/mfdb/models.py` (preferred for first-class values),
   AND check `schema.py` for any CHECK constraint listing allowed values -- update it too.
2. Register at runtime: `db.register_vocabulary_value("artifact_kind", "trace_data", ...)`.

Document which new values were added and where. Do not remove or rename existing values.

### Task 4: Convenience Wrappers

Append to `result_registry.py`:

```python
def register_raw_measurement(
    file_path: str,
    sample_id: str = "",
    metadata: Optional[dict] = None,
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a raw measurement file (TTTR, SPC, BH, ...)."""
    return register_result(
        kind="raw_measurement",
        data=file_path,
        sample_id=sample_id,
        operation_type="measurement_import",
        metadata=metadata,
        db=db,
    )


def register_processed_data(
    data,
    parent_artifact_id: str,
    sample_id: str = "",
    operation_type: str = "analysis",
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
    method: str = "",
    notes: str = "",
    db: Optional["MFDatabase"] = None,
) -> str:
    """Register a calibration result (g-factor, gamma, crosstalk, R0, ...).

    ``method="user_provided"`` (with an empty ``parent_artifact_id``) records a
    value entered from literature or a prior experiment. ``notes`` can carry a
    citation. This wrapper is also consumed by PRD-05; keep the signature stable.
    """
    meta = {"calibration_type": calibration_type}
    if method:
        meta["method"] = method
    if notes:
        meta["notes"] = notes
    return register_result(
        kind="calibration_data",   # add to vocabulary in Task 3
        data=data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type="calibration",  # add to OPERATION_TYPES in Task 3
        parameters=parameters,
        metadata=meta,
        db=db,
    )
```

> `register_calibration` introduces `calibration_data` (artifact kind) and `calibration`
> (operation type). Add both in Task 3. PRD-05 builds directly on this wrapper.

### Task 5: Tests

**File to create**: `test/fio/test_result_registry.py`

Use the **real** fixture pattern (matching `test/fio/test_sample_manager.py`): a plain
`MFDatabase(<tmp>/test.db)` -- the object store root is resolved internally, so do **not**
pass `object_store_root=` and do **not** pre-create an objects directory. Query via
`db.conn` (not `db.con`), and read `mfdb_parameter` columns `name`/`value`.

```python
"""Tests for result_registry.py."""
from __future__ import annotations

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
    set_global_db,
)


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        database = MFDatabase(os.path.join(tmpdir, "test.db"))
        try:
            yield database
        finally:
            database.close()


def test_register_result_with_dict_data(db):
    art_id = register_result(
        kind="processed_data",
        data={"x": [1, 2, 3], "y": [4, 5, 6]},
        operation_type="analysis",
        db=db,
    )
    assert art_id
    row = db.conn.execute(
        "SELECT artifact_kind, object_uuid FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,),
    ).fetchone()
    assert row is not None
    assert row["artifact_kind"] == "processed_data"
    assert row["object_uuid"]  # payload was stored


def test_register_result_with_file(db, tmp_path):
    f = tmp_path / "test.ptu"
    f.write_bytes(b"fake ptu data")
    art_id = register_result(
        kind="raw_measurement",
        data=str(f),
        operation_type="measurement_import",
        db=db,
    )
    assert art_id
    row = db.conn.execute(
        "SELECT data_format FROM mfdb_artifact WHERE artifact_id = ?", (art_id,)
    ).fetchone()
    assert row["data_format"] == "ptu"


def test_register_result_creates_derived_from_edge(db):
    parent_id = register_result(
        kind="raw_measurement", data=b"raw data",
        operation_type="measurement_import", db=db,
    )
    child_id = register_result(
        kind="processed_data", data=b"processed",
        parent_artifact_id=parent_id, operation_type="analysis", db=db,
    )
    row = db.conn.execute(
        """SELECT relationship_type FROM mfdb_edge
           WHERE source_node_id = ? AND target_node_id = ? AND deleted_at IS NULL""",
        (child_id, parent_id),
    ).fetchone()
    assert row is not None
    assert row["relationship_type"] == "derived_from"


def test_register_result_input_link(db):
    parent_id = register_result(
        kind="raw_measurement", data=b"raw",
        operation_type="measurement_import", db=db,
    )
    child_id = register_result(
        kind="processed_data", data=b"proc",
        parent_artifact_id=parent_id, operation_type="analysis", db=db,
    )
    # The child's operation must have an input link to the parent artifact.
    row = db.conn.execute(
        """SELECT oa.direction FROM mfdb_operation_artifact oa
           WHERE oa.artifact_id = ? AND oa.direction = 'input'""",
        (parent_id,),
    ).fetchone()
    assert row is not None


def test_register_result_with_sample(db):
    from chisurf.core.mfdb.models import SampleDefinition
    from chisurf.core.mfdb.sample_manager import create_sample, get_artifacts_for_sample

    sample_id = create_sample(db, SampleDefinition(name="test_sample"))
    art_id = register_result(
        kind="raw_measurement", data=b"data", sample_id=sample_id,
        operation_type="measurement_import", db=db,
    )
    assert art_id in get_artifacts_for_sample(db, sample_id)


def test_register_result_with_parameters(db):
    art_id = register_result(
        kind="fit_result",
        data={"chi2": 1.05},
        operation_type="local_fit",
        parameters={
            "tau1": {"value": 4.0, "error": 0.1, "fixed": False, "bounds": [0, 20], "units": "ns"},
            "amplitude": 0.85,
        },
        db=db,
    )
    rows = db.conn.execute(
        """SELECT p.name, p.value FROM mfdb_parameter p
           JOIN mfdb_operation_artifact oa ON oa.operation_id = p.operation_id
           WHERE oa.artifact_id = ? AND oa.direction = 'output'""",
        (art_id,),
    ).fetchall()
    names = {r["name"] for r in rows}
    assert {"tau1", "amplitude"} <= names


def test_object_store_dedup(db):
    """Identical payloads share one object (refcount increment)."""
    a = register_result(kind="processed_data", data=b"same", operation_type="analysis", db=db)
    b = register_result(kind="processed_data", data=b"same", operation_type="analysis", db=db)
    ua = db.conn.execute("SELECT object_uuid FROM mfdb_artifact WHERE artifact_id=?", (a,)).fetchone()[0]
    ub = db.conn.execute("SELECT object_uuid FROM mfdb_artifact WHERE artifact_id=?", (b,)).fetchone()[0]
    assert ua == ub  # content-addressed dedup


def test_metadata_only_artifact(db):
    art_id = register_result(kind="processed_data", data=None, metadata={"note": "x"}, db=db)
    assert art_id
    row = db.conn.execute(
        "SELECT object_uuid, metadata_json FROM mfdb_artifact WHERE artifact_id=?", (art_id,)
    ).fetchone()
    assert row["object_uuid"] is None
    assert "note" in (row["metadata_json"] or "")


def test_no_db_returns_empty(monkeypatch):
    """With no resolvable DB, register_result returns '' and never raises."""
    import chisurf.core.mfdb.result_registry as rr
    set_global_db(None)
    monkeypatch.setattr(rr, "_get_global_db", lambda: None)
    assert register_result(kind="processed_data", data=b"data") == ""


def test_convenience_wrappers(db, tmp_path):
    f = tmp_path / "m.ptu"
    f.write_bytes(b"raw")
    raw = register_raw_measurement(str(f), db=db, metadata={"k": "v"})
    fit = register_fit_result({"chi2": 1.0}, parent_artifact_id=raw, db=db)
    cal = register_calibration(
        {"g_factor": 1.02}, calibration_type="g_factor",
        method="user_provided", notes="Hellenkamp 2018", db=db,
    )
    assert raw and fit and cal
    row = db.conn.execute(
        "SELECT metadata_json FROM mfdb_artifact WHERE artifact_id=?", (cal,)
    ).fetchone()
    assert "g_factor" in (row["metadata_json"] or "")
```

> When wiring `register_calibration`, ensure `calibration_data`/`calibration` are added in
> Task 3 or the test will fail vocabulary validation for the operation type.

### Task 6: One Reference Integration (prove the API in a real plugin)

Pick one high-value, low-risk producer and wire it as the worked example the rest of
PRD-07 will follow. Recommended: the FCS correlator (`chisurf/plugins/fcs/fcs_correlator/`)
or the burst selection output. After it computes its result:

```python
try:
    from chisurf.core.mfdb.result_registry import register_result
    register_result(
        kind="fcs_correlation",
        data=correlation_dataframe,
        sample_id=sample_id,            # if known, else ""
        parent_artifact_id=source_id,   # the raw measurement artifact, if known
        operation_type="fcs_correlation",
        parameters={"n_casc": n_casc, "n_bins": n_bins},
        metadata={"correlator": "software"},
    )
except Exception:
    pass  # MFDB is optional; never break the plugin
```

This validates the payload archive path inside a real plugin (object store + artifact +
operation + output link) and gives PRD-07 a template. It does **not** prove full
sample/source provenance unless the plugin can pass a real `sample_id` and
`parent_artifact_id`; until those contracts exist, the FCS reference integration is
payload-only provenance.

## Non-Goals / Boundaries

- **Not** a replacement for `project_archiver.py`. That archives whole projects; this
  registers individual plugin outputs. They share repository primitives.
- **Not** responsible for sample creation. Callers pass an existing `sample_id`; sample
  definition is PRD-02's job.
- **No** new threading model. SQLite + auto-commit is the existing contract; hot loops
  pass an explicit `db=`.

## Definition of Done

- [ ] **PRD-030 is complete** and its codec registry exists; `_store_data()` and the
      read path call PRD-030 codecs (msgpack), not `to_json()`/`json.dumps()`
- [ ] `result_registry.py` exists with `register_result()` + 4 convenience wrappers
- [ ] All calls use the **verified** repository methods (`put_object`, `register_artifact`,
      `record_operation`, `record_operation_link`, `record_parameter`, `add_edge`) with
      correct argument names; no invented methods; no `db.con`; no manual `commit()`
- [ ] `_get_global_db()` resolves: explicit override → connector handle → fresh user DB;
      `set_global_db()` override exists
- [ ] Object payloads stored via `put_object` with content-addressed dedup working
- [ ] Provenance correct: `derived_from` edge + `input` operation link for parents;
      `measured_sample` edge for samples; `output` link for the result
- [ ] Parameters written to `mfdb_parameter` with `name`/`value` columns
- [ ] Vocabulary reconciled (Task 3): existing names reused; genuinely new values
      (`trace_data`, `image_data`, `background_data`, `calibration_data`, `calibration`,
      and any new operation types) added to `models.py` and any `schema.py` CHECK lists
- [ ] `register_result` never raises into plugin flow; returns `""` when no DB
- [ ] One real plugin wired as the reference integration (Task 6)
- [ ] All tests in `test/fio/test_result_registry.py` pass
```
