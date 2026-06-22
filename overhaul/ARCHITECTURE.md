# Target Architecture

## Core Principle

MFDB is the single source of truth. Every piece of data -- raw files, processed results,
fit parameters, calibrations, sample metadata -- lives in MFDB or is referenced by it.
The `.csp` file is an export format, not a parallel storage system.

## What Changes

### 1. One Save Path, Not Two

**Current**: Two independent save mechanisms:
- `.csp` ZIP archive (full fidelity, no MFDB)
- `archive_project_to_mfdb()` (lossy, broken restore)

**Target**: One save path. `Project.save()` always writes to MFDB. `.csp` export is a
convenience export built FROM MFDB artifacts, not a separate serialization.

### 2. Sample-Centric Data Model

**Current**: Projects contain datasets and fits. No sample association.

**Target**: Every dataset belongs to a sample. Every sample has:
- A name and description
- An entity (protein/DNA sequence with labeling positions)
- Attached probes (donor/acceptor dyes)
- Conditions (buffer, pH, temperature)

The sample is the anchor. When you open mfdb-admin, you see samples, not orphan files.

### 3. Result Registry

**Current**: 79 of 87 plugins produce results that vanish when ChiSurf closes.

**Target**: A simple `ResultRegistry` API that any plugin can call:

```python
from chisurf.core.mfdb.result_registry import register_result

register_result(
    kind="burst_data",            # what type of result
    data=dataframe_or_path,       # the actual data
    sample_id="sm_dna_001",       # which sample (required)
    parent_artifact_id="abc123",  # what input produced this
    operation_type="burst_selection",
    parameters={"min_photons": 50, "time_window_us": 500},
)
```

That's it. One function call. The registry handles:
- Storing data in the object store
- Creating the artifact record
- Creating the operation record
- Creating provenance edges
- Linking to the sample

### 4. Simplified Schema Usage

**Current**: Three generations of tables (legacy FLR, FDB, canonical MFDB). New code
sometimes uses wrong vocabulary constants.

**Target**: New code uses only these tables:

| Table | Purpose |
|-------|---------|
| `mfdb_object` | Content-addressed blob storage |
| `mfdb_artifact` | Data artifacts (files, results, sessions) |
| `mfdb_operation` | Processing steps (measurement, fit, calibration) |
| `mfdb_operation_artifact` | Links operations to their inputs/outputs |
| `mfdb_edge` | Provenance relationships |
| `mfdb_parameter` | Fit/analysis parameters with statistics |
| `mfdb_sample` | Sample definitions |
| `mfdb_experiment` | Experiment records |
| `mfdb_setup` | Instrument configurations |

Legacy tables remain for migration but no new code writes to them.

### 5. Plugin Data Flow

```
Plugin produces result
       |
       v
ResultRegistry.register_result()
       |
       v
MFDB: artifact + operation + edges + parameters
       |
       v
mfdb-admin / project browser can see it
```

Every plugin that produces data calls `register_result()`. No plugin needs to understand
MFDB internals. The registry is the only integration point.

## Directory Structure (changes only)

```
chisurf/core/mfdb/
    result_registry.py     # NEW: simple registration API
    sample_manager.py      # NEW: sample CRUD with validation
    schema.py              # existing, no changes
    models.py              # existing, add SampleDefinition dataclass
    repository.py          # existing, add sample query methods
    project_archiver.py    # REWRITE: use result registry internally
    pipeline.py            # REWRITE: use result registry, fix vocabulary
    chinet_adapter.py      # existing, minor fixes
```

## Vocabulary Constants (Canonical)

These are the ONLY values new code should use. They are defined in `models.py`.

**artifact_kind**: `raw_measurement`, `processed_data`, `burst_data`, `fit_result`,
`calibration_data`, `chinet_session`, `chinet_node`, `project_metadata`, `spectrum_data`,
`irf_data`, `background_data`

**operation_type**: `measurement`, `burst_selection`, `histogram_construction`,
`background_correction`, `local_fit`, `global_fit`, `calibration`, `project`,
`population_selection`, `correlation`, `image_analysis`

**storage_mode**: `object_store`, `local_file`, `inline_json`, `external_reference`

**relationship_type**: `derived_from`, `calibrated_by`, `parameter_depends_on`,
`project_contains`, `supersedes`, `measured_sample`, `used_setup`, `references`

**status**: `succeeded`, `failed`, `running`, `pending`
