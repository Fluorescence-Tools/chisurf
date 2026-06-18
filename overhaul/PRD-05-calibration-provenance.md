# PRD-05: Calibration Provenance

## Goal

Calibration parameters (g-factor, gamma, crosstalk, direct excitation, Forster radius)
are tracked in MFDB with links to the reference measurements they came from. When a
calibration value changes, all downstream fits that used it can be identified.

## Background

Read these files before starting:
- `chisurf/core/models/tcspc/lifetime.py` -- search for `background_curve`, `scatter`, `t_bg`, `t_exp`
- `chisurf/core/models/tcspc/anisotropy.py` -- search for `g_factor`, `l1`, `l2`
- `chisurf/core/models/tcspc/fret.py` -- search for `R0`, `tauD0`, `kappa2`
- `chisurf/core/models/pda/nusiance.py` -- search for `crosstalk`, `gamma`, `direct_excitation`
- `chisurf/core/fluorescence/fret/__init__.py` -- intensity-based FRET corrections
- `chisurf/plugins/jordi_g_factor/` -- g-factor calculator plugin
- `chisurf/core/mfdb/result_registry.py` -- from PRD-03

## What Are Calibration Parameters

In smFRET, before you can get real distances, you need these correction parameters:

| Parameter | What | How Measured |
|-----------|------|-------------|
| g-factor | Detector polarization sensitivity ratio | Measure fast-rotating dye (e.g. Rhodamine 110), tail-match VV/VH |
| gamma | Detection efficiency ratio (donor/acceptor channels) | Donor-only sample + DA sample comparison |
| crosstalk (alpha) | Donor leakage into acceptor channel | Donor-only measurement, count ratio |
| Direct excitation (delta) | Acceptor excited by donor laser | Acceptor-only measurement |
| tau_D0 | Donor-only lifetime | Fit donor-only decay |
| R0 | Forster radius | Computed from spectra (see PRD-06) |

Currently these are free `FittingParameter` objects. Nobody tracks where their values
came from.

## Tasks

### Task 1: Define Calibration Data Model

**File**: `chisurf/core/mfdb/models.py`

Add a dataclass for calibration records:

```python
@dataclasses.dataclass
class CalibrationRecord:
    """A calibration measurement and its derived parameter values.

    Calibration values come from two sources:
    1. Derived from a reference measurement (method="tail_matching", "intensity_ratio", etc.)
       -> source_artifact_id points to the reference measurement artifact
    2. Entered manually by the user without backing data (method="user_provided")
       -> source_artifact_id is empty, source_file is empty

    Both paths are valid. "God given" values (e.g. R0 from literature) are stored
    with method="user_provided" and optionally a note citing the source.
    """
    calibration_type: str  # "g_factor", "gamma", "crosstalk", "direct_excitation", "donor_lifetime", "forster_radius"
    value: float
    error: float = 0.0
    source_file: str = ""          # path to reference measurement file (empty for user_provided)
    source_artifact_id: str = ""   # MFDB artifact of the reference measurement (empty for user_provided)
    sample_id: str = ""            # which sample (for donor-only, acceptor-only refs)
    method: str = ""               # "tail_matching", "intensity_ratio", "spectral_overlap", "user_provided"
    notes: str = ""                # for user_provided: cite literature source, e.g. "Hellenkamp et al. 2018"
```

### Task 2: Register Calibrations When Computed

#### g-factor (jordi_g_factor plugin)

**File**: `chisurf/plugins/jordi_g_factor/` (find the main calculation file)

After the g-factor is computed, register it:

```python
from chisurf.core.mfdb.result_registry import register_calibration

register_calibration(
    data={"g_factor": computed_g_factor, "tail_start_ns": tail_start},
    calibration_type="g_factor",
    parameters={"g_factor": {"value": computed_g_factor, "error": g_error}},
    parent_artifact_id=reference_artifact_id,  # the fast-rotating dye measurement
)
```

**How to find the right place**: Look for where the g-factor value is computed and
displayed to the user. That's where you add the registration call. Wrap in try/except
so the plugin still works without MFDB.

#### Donor lifetime (lltf plugin or lifetime model)

**File**: `chisurf/plugins/fluorescence_decay/lltf/` (find the fit execution)

After a fit completes, if the model is a simple lifetime model and the dataset is tagged
as "donor-only" (via sample metadata), register tau_D0:

```python
# After fit completes:
if fit.model.name == "Lifetime" and is_donor_only:
    from chisurf.core.mfdb.result_registry import register_calibration
    tau_values = fit.model.lifetime_spectrum  # get the lifetimes
    register_calibration(
        data={"tau_D0": tau_values[0], "amplitudes": amplitude_values},
        calibration_type="donor_lifetime",
        parameters={"tau_D0": {"value": tau_values[0], "error": tau_error}},
        parent_artifact_id=dataset_artifact_id,
    )
```

**Note**: The "is_donor_only" check can be done by looking at the sample metadata.
If the sample has a donor probe but no acceptor probe, it's donor-only.

#### Crosstalk and direct excitation

These are typically entered manually by the user or computed from intensity ratios.
For now, provide a way to register them when set in the PDA nuisance widget.

**File**: `chisurf/gui/widgets/models/` (find the PDA nuisance parameter widget)

Add a "Save to MFDB" button next to the crosstalk/gamma/direct_excitation fields.
When clicked:

```python
register_calibration(
    data={"crosstalk": alpha_value},
    calibration_type="crosstalk",
    parameters={"crosstalk": alpha_value},
)
```

### Task 2b: Register "God Given" Calibration Values

Not all calibration parameters come from reference measurements. Users often enter
values from literature (e.g., R0 = 54 A for Alexa488-Alexa647 from Hellenkamp et al.)
or from prior experiments. These must also be tracked in MFDB.

**File**: `chisurf/core/mfdb/result_registry.py`

The existing `register_calibration()` function (from PRD-03) already works for this.
The key is `method="user_provided"` and an empty `parent_artifact_id`:

```python
# Example: user enters R0 directly
register_calibration(
    data={"R0": 54.0},
    calibration_type="forster_radius",
    parameters={"R0": {"value": 54.0, "error": 2.0}},
    parent_artifact_id="",  # no reference measurement
    db=db,
)
```

The `register_calibration()` wrapper stores `calibration_type` in metadata. To
distinguish user-provided from derived values, add `method` to the metadata:

Update `register_calibration()` in `result_registry.py`:

```python
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
    """Register a calibration result (g-factor, gamma, crosstalk, R0, etc.).

    Args:
        method: How the value was determined. Use "user_provided" for values
            entered manually (literature values, prior experiments).
            Other values: "tail_matching", "intensity_ratio", "spectral_overlap".
        notes: Free-text note, e.g. literature citation for user_provided values.
    """
    meta = {"calibration_type": calibration_type}
    if method:
        meta["method"] = method
    if notes:
        meta["notes"] = notes
    return register_result(
        kind="calibration_data",
        data=data,
        sample_id=sample_id,
        parent_artifact_id=parent_artifact_id,
        operation_type="calibration",
        parameters=parameters,
        metadata=meta,
        db=db,
    )
```

**GUI side**: In any widget that has calibration parameter fields (g-factor, gamma,
crosstalk, R0), add a "Save to MFDB" button. When the user clicks it without having
run a reference measurement, use `method="user_provided"`. The button should show a
small text input for an optional citation/note.

### Task 3: Link Fits to Their Calibration Sources

When a fit is archived to MFDB (via `project_archiver.py` or `chinet_adapter.py`),
check if any of its parameters match known calibration records.

**File**: `chisurf/core/mfdb/project_archiver.py`

In `_archive_fits()`, after creating the fit operation and recording parameters, query
for calibration artifacts that match:

```python
def _link_calibrations(db, fit_operation_id, fit_parameters):
    """Link a fit to calibration sources via 'calibrated_by' edges.

    For each calibration parameter in the fit (g_factor, gamma, crosstalk,
    direct_excitation, R0, tau_D0), look for a calibration artifact with a
    matching value. If found, create a 'calibrated_by' edge.
    """
    calibration_param_names = {
        "g_factor", "gamma", "crosstalk", "direct_excitation",
        "delta", "alpha", "R0", "forster_radius", "tauD0", "tau_D0",
    }

    for param_name, param_value in fit_parameters.items():
        if param_name.lower().replace("-", "_") not in calibration_param_names:
            continue

        # Look for calibration artifacts with matching value
        rows = db.con.execute(
            """SELECT p.operation_id, oa.artifact_id
               FROM mfdb_parameter p
               JOIN mfdb_operation o ON o.operation_id = p.operation_id
               JOIN mfdb_operation_artifact oa ON oa.operation_id = p.operation_id
               WHERE o.operation_type = 'calibration'
               AND p.parameter_name = ?
               ORDER BY o.created_at DESC
               LIMIT 1""",
            (param_name,)
        ).fetchone()

        if rows:
            cal_artifact_id = rows[1]
            db.add_edge(
                source_id=fit_operation_id,
                source_type="operation",
                target_id=cal_artifact_id,
                target_type="artifact",
                relationship_type="calibrated_by",
            )
```

Call this after recording the fit's parameters.

### Task 4: Link Background Curves to MFDB

**File**: `chisurf/core/mfdb/project_archiver.py`

In `_archive_fits()`, check if the fit state includes a background curve. If it does,
register the background file as an artifact and create a `calibrated_by` edge.

```python
# In _archive_fits(), for each fit:
bg_path = fit_state.get("background_curve_path", "")
if bg_path and os.path.isfile(bg_path):
    from chisurf.core.mfdb.result_registry import register_raw_measurement
    bg_art_id = register_raw_measurement(
        file_path=bg_path,
        sample_id=sample_id,
        metadata={"purpose": "background_reference"},
        db=db,
    )
    db.add_edge(
        source_id=fit_operation_id,
        source_type="operation",
        target_id=bg_art_id,
        target_type="artifact",
        relationship_type="calibrated_by",
    )
```

**Prerequisite**: The fit state must include the background curve file path. Check
`chisurf/core/project/fit_state.py` -- in `_model_to_state()`, verify that the
background curve path is included. If not, add it:

```python
# In _model_to_state(), find where Generic/nuisance parameters are serialized:
if hasattr(model, 'generic') and hasattr(model.generic, 'background_curve'):
    bg = model.generic.background_curve
    if bg is not None and hasattr(bg, 'filename'):
        state["background_curve_path"] = bg.filename
```

### Task 5: Staleness Detection Query

**File to create**: `chisurf/core/mfdb/staleness.py`

```python
"""Detect stale analysis results when calibrations change.

A fit is "stale" if:
1. It has a 'calibrated_by' edge to a calibration artifact
2. A newer calibration artifact of the same type exists
"""
from chisurf.core.mfdb.repository import MFDatabase


def find_stale_fits(db: MFDatabase) -> list:
    """Find fits that use outdated calibration values.

    Returns list of dicts:
    [
        {
            "fit_operation_id": "...",
            "calibration_type": "g_factor",
            "used_artifact_id": "old_cal_123",
            "latest_artifact_id": "new_cal_456",
            "used_value": 1.02,
            "latest_value": 1.05,
        },
        ...
    ]
    """
    stale = []

    # Find all fits with calibrated_by edges
    rows = db.con.execute(
        """SELECT e.source_id AS fit_op_id,
                  e.target_id AS cal_artifact_id,
                  a.metadata_json
           FROM mfdb_edge e
           JOIN mfdb_artifact a ON a.artifact_id = e.target_id
           WHERE e.relationship_type = 'calibrated_by'"""
    ).fetchall()

    for row in rows:
        fit_op_id = row[0]
        cal_art_id = row[1]
        meta = row[2]
        if not meta:
            continue

        import json
        meta_dict = json.loads(meta) if isinstance(meta, str) else meta
        cal_type = meta_dict.get("calibration_type", "")
        if not cal_type:
            continue

        # Find the latest calibration of this type
        latest = db.con.execute(
            """SELECT oa.artifact_id, p.parameter_value
               FROM mfdb_operation o
               JOIN mfdb_operation_artifact oa ON oa.operation_id = o.operation_id
               JOIN mfdb_parameter p ON p.operation_id = o.operation_id
               WHERE o.operation_type = 'calibration'
               AND oa.direction = 'output'
               ORDER BY o.created_at DESC
               LIMIT 1"""
        ).fetchone()

        if latest and latest[0] != cal_art_id:
            # Get the value used by the fit
            used_param = db.con.execute(
                """SELECT p.parameter_value FROM mfdb_parameter p
                   WHERE p.operation_id = ?
                   LIMIT 1""",
                (fit_op_id,)
            ).fetchone()

            stale.append({
                "fit_operation_id": fit_op_id,
                "calibration_type": cal_type,
                "used_artifact_id": cal_art_id,
                "latest_artifact_id": latest[0],
                "used_value": used_param[0] if used_param else None,
                "latest_value": latest[1],
            })

    return stale
```

### Task 6: Write Tests

**File to create**: `test/fio/test_calibration_provenance.py`

```python
"""Tests for calibration provenance."""
import os
import tempfile
import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.result_registry import register_result, register_calibration


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        obj_root = os.path.join(tmpdir, "objects")
        os.makedirs(obj_root)
        database = MFDatabase(db_path, object_store_root=obj_root)
        yield database
        database.close()


def test_register_calibration_creates_artifact(db):
    art_id = register_calibration(
        data={"g_factor": 1.02},
        calibration_type="g_factor",
        parameters={"g_factor": {"value": 1.02, "error": 0.01}},
        db=db,
    )
    assert art_id

    row = db.con.execute(
        "SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,)
    ).fetchone()
    assert row[0] == "calibration_data"


def test_calibration_linked_to_reference(db, tmp_path):
    # Create a reference measurement
    ref_file = tmp_path / "rh110.ptu"
    ref_file.write_bytes(b"reference data")
    ref_id = register_result(
        kind="raw_measurement",
        data=str(ref_file),
        operation_type="measurement",
        db=db,
    )

    # Create calibration derived from reference
    cal_id = register_calibration(
        data={"g_factor": 1.02},
        calibration_type="g_factor",
        parent_artifact_id=ref_id,
        parameters={"g_factor": 1.02},
        db=db,
    )

    # Check derived_from edge
    row = db.con.execute(
        """SELECT relationship_type FROM mfdb_edge
           WHERE source_id = ? AND target_id = ?""",
        (cal_id, ref_id)
    ).fetchone()
    assert row is not None
    assert row[0] == "derived_from"
```

## Definition of Done

- [ ] `CalibrationRecord` dataclass exists in `models.py` with `method` field supporting `"user_provided"`
- [ ] g-factor plugin registers calibrations in MFDB
- [ ] `register_calibration()` supports `method` and `notes` parameters for manual entry
- [ ] User-provided calibrations (R0 from literature, etc.) can be stored without a parent artifact
- [ ] Fit archiver links fits to calibration sources via `calibrated_by` edges
- [ ] Background curve files are registered as artifacts
- [ ] `staleness.py` can find fits using outdated calibrations
- [ ] All tests pass (including test for user_provided calibration)
