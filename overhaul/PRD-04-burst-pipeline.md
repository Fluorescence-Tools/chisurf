# PRD-04: Connect Burst Pipeline to MFDB

## Goal

When the burst selection plugin produces results, those results are automatically
registered in MFDB with provenance linking back to the raw TTTR file and forward to
any downstream analysis.

## Background

Read these files before starting:
- `chisurf/core/mfdb/pipeline.py` -- current burst pipeline (broken vocabulary)
- `chisurf/plugins/burst/burst_selection/api/selection.py` -- burst selection logic
- `chisurf/plugins/burst/burst_selection/api/models.py` -- data models
- `chisurf/plugins/burst/burst_selection/api/contract.py` -- RPC contract
- `chisurf/core/mfdb/result_registry.py` -- the result registry from PRD-03

## Current Problems

1. `pipeline.py` uses wrong vocabulary:
   - `artifact_type` should be `artifact_kind`
   - `storage_mode="local"` should be `"local_file"`
   - `status="success"` should be `"succeeded"`
2. `pipeline.py` is standalone -- not connected to project archiver
3. Burst selection plugin (`plugins/burst/burst_selection/`) has no MFDB code at all
4. No burst results appear in mfdb-admin

## Tasks

### Task 1: Fix pipeline.py Vocabulary

**File**: `chisurf/core/mfdb/pipeline.py`

Find and replace these exact strings:

1. Find `artifact_type` and replace with `artifact_kind` (all occurrences)
2. Find `storage_mode="local"` and replace with `storage_mode="local_file"`
3. Find `status="success"` and replace with `status="succeeded"`
4. Find `"success"` used as a status value and replace with `"succeeded"`

After fixing, verify by searching for any remaining uses of the old values.

### Task 2: Rewrite pipeline.py to Use Result Registry

**File**: `chisurf/core/mfdb/pipeline.py`

The current `BurstPipeline` class does too much manually. Replace with calls to the
result registry.

The class has roughly this structure (verify by reading the file):

```python
class BurstPipeline:
    def __init__(self, db, ...):
        ...
    def register_input(self, file_path, ...):
        # Creates raw_measurement artifact
        ...
    def register_output(self, file_path, ...):
        # Creates processed data artifact
        ...
```

Rewrite to use `register_result()`:

```python
"""Burst pipeline MFDB integration.

Registers burst selection inputs and outputs in MFDB via the result registry.
"""
from chisurf.core.mfdb.result_registry import (
    register_raw_measurement,
    register_result,
)


class BurstPipeline:
    """Registers burst selection workflow steps in MFDB."""

    def __init__(self, db=None, sample_id: str = ""):
        self.db = db
        self.sample_id = sample_id
        self._input_artifact_ids = []

    def register_input(self, file_path: str, metadata: dict = None) -> str:
        """Register a raw TTTR file as input.

        Args:
            file_path: Path to the TTTR file (.ptu, .ht3, .spc, etc.)
            metadata: Optional metadata (header info, etc.)

        Returns:
            artifact_id of the registered input.
        """
        art_id = register_raw_measurement(
            file_path=file_path,
            sample_id=self.sample_id,
            metadata=metadata,
            db=self.db,
        )
        self._input_artifact_ids.append(art_id)
        return art_id

    def register_burst_selection(
        self,
        output_path: str,
        detection_settings: dict,
        filter_settings: dict = None,
        burst_count: int = 0,
    ) -> str:
        """Register burst selection results.

        Args:
            output_path: Path to the .bur output file.
            detection_settings: Dict with detection parameters:
                {"algorithm": "sliding_window", "min_photons": 50,
                 "time_window_us": 500, "photon_window": 500}
            filter_settings: Optional filter parameters.
            burst_count: Number of bursts detected.

        Returns:
            artifact_id of the burst data artifact.
        """
        parent_id = self._input_artifact_ids[0] if self._input_artifact_ids else ""

        metadata = {
            "detection_settings": detection_settings,
            "filter_settings": filter_settings or {},
            "burst_count": burst_count,
        }

        art_id = register_result(
            kind="burst_data",
            data=output_path,
            sample_id=self.sample_id,
            parent_artifact_id=parent_id,
            operation_type="burst_selection",
            parameters=_extract_parameters(detection_settings),
            metadata=metadata,
            db=self.db,
        )
        return art_id


def _extract_parameters(settings: dict) -> dict:
    """Extract numeric parameters from detection settings for mfdb_parameter."""
    params = {}
    for key in ("min_photons", "time_window_us", "photon_window", "threshold"):
        if key in settings and isinstance(settings[key], (int, float)):
            params[key] = settings[key]
    return params
```

### Task 3: Call Pipeline from Burst Selection Plugin

**File**: `chisurf/plugins/burst/burst_selection/api/selection.py`

Find the main function that runs burst selection (likely `analyze_files()` or similar).
After it produces the output `.bur` file, add a call to register the result.

Look for where the output file path is created. After that point, add:

```python
# Register in MFDB
try:
    from chisurf.core.mfdb.pipeline import BurstPipeline

    pipeline = BurstPipeline(sample_id=request.sample_id if hasattr(request, 'sample_id') else "")
    for input_file in input_files:
        pipeline.register_input(input_file)
    pipeline.register_burst_selection(
        output_path=output_bur_path,
        detection_settings={
            "algorithm": request.burst_detection.algorithm,
            "min_photons": request.burst_detection.min_photons,
            "time_window_us": request.burst_detection.time_window,
            "photon_window": request.burst_detection.photon_window,
        },
        burst_count=len(burst_dataframe),
    )
except Exception:
    import logging
    logging.getLogger(__name__).warning("Failed to register burst results in MFDB", exc_info=True)
```

**Important**: Read the actual selection.py to find:
1. The function that runs the analysis
2. The variable names for input files, output path, burst count
3. The request object structure (from `models.py`)

Adapt the code above to use the actual variable names.

### Task 4: Add sample_id to AnalysisRequest

**File**: `chisurf/plugins/burst/burst_selection/api/models.py`

Find the `AnalysisRequest` dataclass and add a `sample_id` field:

```python
@dataclasses.dataclass
class AnalysisRequest:
    # ... existing fields ...
    sample_id: str = ""  # MFDB sample to link results to
```

### Task 5: Add Sample Picker to Burst Selection GUI

**File**: `chisurf/plugins/burst/burst_selection/gui/` (find the main widget file)

Add a `SamplePicker` widget (from PRD-02) to the burst selection GUI. When the user
runs burst selection, the selected `sample_id` is passed to `AnalysisRequest`.

Look at the GUI code to find where the "Run" button is connected. The `sample_id` from
the picker should be included in the request.

### Task 6: Write Tests

**File to create**: `test/fio/test_burst_pipeline_mfdb.py`

```python
"""Test burst pipeline MFDB integration."""
import os
import tempfile
import pytest

from chisurf.core.mfdb.repository import MFDatabase
from chisurf.core.mfdb.pipeline import BurstPipeline
from chisurf.core.mfdb.sample_manager import create_sample
from chisurf.core.mfdb.models import SampleDefinition


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        obj_root = os.path.join(tmpdir, "objects")
        os.makedirs(obj_root)
        database = MFDatabase(db_path, object_store_root=obj_root)
        yield database
        database.close()


@pytest.fixture
def sample_id(db):
    return create_sample(db, SampleDefinition(name="test_burst_sample"))


def test_register_input(db, tmp_path, sample_id):
    # Create dummy input file
    f = tmp_path / "test.spc"
    f.write_bytes(b"fake spc data")

    pipeline = BurstPipeline(db=db, sample_id=sample_id)
    art_id = pipeline.register_input(str(f))

    assert art_id
    row = db.con.execute(
        "SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,)
    ).fetchone()
    assert row[0] == "raw_measurement"


def test_register_burst_selection(db, tmp_path, sample_id):
    # Create dummy files
    input_f = tmp_path / "test.spc"
    input_f.write_bytes(b"fake spc")
    output_f = tmp_path / "test.bur"
    output_f.write_bytes(b"burst data")

    pipeline = BurstPipeline(db=db, sample_id=sample_id)
    pipeline.register_input(str(input_f))
    art_id = pipeline.register_burst_selection(
        output_path=str(output_f),
        detection_settings={"algorithm": "sliding_window", "min_photons": 50},
        burst_count=1000,
    )

    assert art_id
    # Check artifact type
    row = db.con.execute(
        "SELECT artifact_kind FROM mfdb_artifact WHERE artifact_id = ?",
        (art_id,)
    ).fetchone()
    assert row[0] == "burst_data"

    # Check derived_from edge exists
    row = db.con.execute(
        """SELECT relationship_type FROM mfdb_edge
           WHERE source_id = ? AND relationship_type = 'derived_from'""",
        (art_id,)
    ).fetchone()
    assert row is not None

    # Check parameters were recorded
    rows = db.con.execute(
        """SELECT parameter_name, parameter_value FROM mfdb_parameter
           WHERE operation_id IN (
               SELECT operation_id FROM mfdb_operation_artifact
               WHERE artifact_id = ? AND direction = 'output'
           )""",
        (art_id,)
    ).fetchall()
    param_names = {r[0] for r in rows}
    assert "min_photons" in param_names


def test_burst_pipeline_links_to_sample(db, tmp_path, sample_id):
    input_f = tmp_path / "test.spc"
    input_f.write_bytes(b"fake spc")

    pipeline = BurstPipeline(db=db, sample_id=sample_id)
    art_id = pipeline.register_input(str(input_f))

    # Check measured_sample edge
    row = db.con.execute(
        """SELECT target_id FROM mfdb_edge
           WHERE source_id = ? AND relationship_type = 'measured_sample'""",
        (art_id,)
    ).fetchone()
    assert row is not None
    assert row[0] == sample_id
```

## Definition of Done

- [ ] `pipeline.py` uses correct vocabulary (artifact_kind, local_file, succeeded)
- [ ] `pipeline.py` uses `register_result()` from the result registry
- [ ] Burst selection plugin calls pipeline after producing results
- [ ] `AnalysisRequest` has a `sample_id` field
- [ ] Burst selection GUI has a sample picker
- [ ] Burst results appear in mfdb-admin with provenance edges
- [ ] All tests pass
