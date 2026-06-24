# Burst Selection Reference Implementation

Burst Selection is the reference implementation for workflow-ready ChiSurf
plugins. It shows how a plugin should separate scientific code, transport
adapters, GUI code, and MFDB archival side effects.

The important rule is ownership: the plugin owns its workflow and MFDB
integration. `chisurf.core` provides generic MFDB primitives only. Core must not
import a plugin backend, run plugin analysis, or hand-write plugin-specific
artifact rows.

## Layer Responsibilities

| Layer | Files | Responsibility |
| --- | --- | --- |
| Models | `api/models.py` | Dataclass request, settings, result, and MFDB context types. |
| Contract | `api/contract.py` | JSON-safe payload normalization, canonical RPC method names, contract descriptor. |
| Pure analysis | `api/selection.py`, `api/features.py`, `api/io.py` | Compute burst tables and write requested analysis files. No MFDB, no Qt, no RPC transport. |
| MFDB archival | `api/mfdb.py` | Register successful analysis inputs and outputs with MFDB using result-registry APIs. |
| Service adapter | `backend/services.py` | Accept JSON/RPC params, build `AnalysisRequest`, run pure analysis, then run MFDB archival. |
| Client | `gui/client.py`, `server/client.py` | Send canonical method calls and JSON-safe payloads. |
| GUI | `gui/tool.py` | Build request context from controls and render results. It does not own analysis behavior. |
| Tests | `tests/`, `test/fio/test_burst_pipeline_mfdb.py` | Verify API contract, plugin behavior, and MFDB provenance. |

Future plugins should copy this split before copying individual code.

## Request Flow

All user-facing entry points normalize into the same request object:

```text
GUI / CLI / RPC / Python caller
  -> JSON-safe payload
  -> analysis_request_from_payload()
  -> AnalysisRequest
  -> analyze_request()
  -> AnalysisResult
```

`AnalysisRequest` carries all scientific and archival inputs:

- `files`: input TTTR paths.
- `filetype`, `windows`, `detectors`, `selected_setup`: reader and detector setup context.
- `settings`: photon filter, burst detection, GMM, and output settings.
- `output_dir`, `legacy_output`, `legacy_output_folder_name`: output file layout.
- `mfdb`: nested `MFDBContext`.

MFDB context is intentionally nested:

```json
{
  "mfdb": {
    "enabled": true,
    "sample_id": "sample_1",
    "source_artifact_ids": {
      "/absolute/input/m000.spc": "existing_raw_artifact"
    },
    "register_missing_inputs": true
  }
}
```

Do not add top-level request fields for provenance. Add future archival fields
inside `MFDBContext` so the request boundary stays stable.

## GUI MFDB Output Preflight

The GUI exposes MFDB as an output mode alongside file outputs such as CSV
(`.bur`) and HDF5. Selecting MFDB does not remove the other output choices;
MFDB-only runs are valid, while zip/remove-folder controls remain tied to file
outputs.

Before batch processing starts with MFDB selected, `gui/tool.py` resolves each
raw TTTR file to its content MD5 and asks MFDB whether that content is already
linked to a sample. The sample represents the molecular definition of the
measurement: host molecule, attached dyes, buffer/conditions, and related
metadata.

If every raw input is already registered, the GUI reuses the existing sample and
raw artifacts in the nested `mfdb` request context. If a raw input has no sample
association, the GUI opens the shared sample registration dialog
(`show_sample_picker_dialog`) before analysis. Cancelling that dialog aborts the
MFDB run before any burst processing starts. Completing it registers or selects a
sample, registers missing raw inputs as `raw_measurement` artifacts, writes the
sample ID onto the object-store reference, and passes those artifact IDs in
`mfdb.source_artifact_ids`.

Selected-file previews and filter-refresh diagnostics do not perform MFDB
archival. Only the explicit batch process action with MFDB selected owns this
preflight.

## Setup Storage

Detector and PIE-window definitions are MFDB setup definitions. The detector
wizard persists setups to `mfdb_setup` via `load_detector_setups()` and
`save_detector_setups()`; the old `detector_setups.json` file is imported only
as a migration/fallback source. The setup row stores the original wizard payload
in `configuration_json`, detector routing in `detectors_json`, TTTR timing in
`timing_resolution_json`, and burst defaults in `burst_defaults_json`.

Setup IDs are deterministic from the setup name, for example
`tttr_detector_setup:bh_spc_130`. GUI MFDB preflight ensures the selected setup
row exists before processing starts and puts that ID in `mfdb.setup_id`.
Result registration passes `setup_id` to `register_result()`, so the
`measurement_import` and `burst_selection` operations point at the setup through
`mfdb_operation.setup_id`. Consumers should use the operation setup link rather
than parsing detector definitions from artifact metadata.

## Pure Analysis Contract

`analyze_request(request)` is pure with respect to MFDB. It may read TTTR files
and write requested analysis outputs such as `.bur`, HDF5, zip files, or legacy
output folders, but it does not require a database.

This keeps scientific behavior directly testable:

```python
from chisurf.plugins.burst.burst_selection.api import AnalysisRequest, analyze_request

result = analyze_request(AnalysisRequest(files=["m000.spc"]))
```

The result contains:

- `dataframes`: per-input burst rows as JSON-safe records.
- `output_paths`: compatibility role map for callers that only need the last path
  for a role.
- `output_paths_by_file`: stable per-input role map used for provenance.
- `metadata`: run counts and analysis metadata.
- `mfdb_artifacts`: filled by the service/archival layer, not by pure analysis.
- `warnings`: non-fatal warnings, including MFDB archival warnings.

For multi-file runs, `output_paths_by_file` is the authoritative output map:

```json
{
  "/data/m000.spc": {"bur": "/out/m000.bur"},
  "/data/m001.spc": {"bur": "/out/m001.bur"}
}
```

Future plugins that produce repeated output roles per input should implement the
same pattern instead of using only a flat role map.

## MFDB Archival Flow

MFDB registration happens after pure analysis succeeds:

```text
analyze_files_handler()
  -> analysis_request_from_payload()
  -> analyze_request()
  -> BurstMFDBPipeline().register_run(request, result)
  -> result.mfdb_artifacts + result.warnings
```

The archival graph is:

```text
raw TTTR artifact
  -> operation: burst_selection
  -> burst_table artifact
  -> optional sidecar artifacts
```

`api/mfdb.py` uses PRD-03 result-registry APIs:

- `register_raw_measurement()` for input TTTR files.
- `register_result(kind="burst_table", ...)` for primary burst outputs.
- `register_result(kind="processed_data" | "external_reference", ...)` for sidecars.

It accepts `chisurf.core.mfdb.MFDBClientBase` implementations, not just the
local SQLite `MFDatabase` repository. This keeps plugins compatible with future
remote or in-process MFDB clients that implement the same artifact, object-store,
sample lookup, and provenance methods.

Plugin archival code does not hand-write object-store rows, artifact rows,
operation rows, or parameter rows.

## Input Artifact Rules

For each input file, `BurstMFDBPipeline.register_run()` normalizes the file path
with `Path(path).resolve()`.

If `request.mfdb.source_artifact_ids` contains that normalized path, the pipeline
uses the supplied raw artifact ID. It does not duplicate the raw measurement.

If no source artifact is supplied and `register_missing_inputs` is true, it
registers a new `raw_measurement` artifact:

```python
register_raw_measurement(
    file_path=input_path,
    sample_id=request.mfdb.sample_id,
    metadata={
        "plugin": "burst_selection",
        "role": "raw_tttr",
        "filetype": request.filetype,
        "selected_setup": request.selected_setup,
    },
    db=db,
)
```

If a sample ID is supplied, the result registry links the artifact to that
sample. Invalid sample IDs are reported as warnings and do not leave partial
artifact/object rows.

## Burst Table Rules

The primary scientific output is always a `burst_table` artifact.

Payload choice is deterministic:

1. If a per-file `.bur` path exists in `output_paths_by_file`, register that file
   as a `burst_table` artifact with `data_format="bur"`.
2. If no `.bur` path exists but per-file rows exist in `dataframes`, convert the
   rows to a DataFrame and register a typed msgpack `BurstTable` payload.

Each burst table receives:

- `operation_type="burst_selection"`
- `parent_artifact_id=<matching raw TTTR artifact>`
- a `derived_from` edge to the matching raw input
- scalar parameters in `mfdb_parameter`
- JSON-safe nested settings in metadata

Scalar parameters include:

- `min_photons`
- `photon_window`
- `time_window`
- `filter_active`
- `count_rate_n_ph_max`
- `count_rate_time_window`
- `delta_macro_time_min`
- `delta_macro_time_max`
- `gmm_max_components`

Nested settings are metadata, not parameter rows.

## Sidecar Rules

Sidecars are registered only after at least one primary burst table exists.

| Role | Artifact kind | Parent |
| --- | --- | --- |
| `hdf5` | `processed_data` | first burst table artifact |
| `zip` | `processed_data` | first burst table artifact |
| `output_folder` | `external_reference` or `processed_data` | first burst table artifact |
| `mti_dir` | `external_reference` | matching burst table artifact |

Sidecar metadata includes `plugin`, `contract_version`, `output_role`, and
`path`. Sidecars are convenience outputs; MFDB consumers should treat
`burst_table` artifacts as the primary scientific result.

## Failure Semantics

MFDB archival is best effort in user-facing flows:

- `mfdb.enabled=false`: skip registration and return empty artifact maps.
- Missing DB: keep analysis success and return warnings.
- Invalid `sample_id`: keep analysis success and return warnings.
- Invalid source artifact ID: skip that input and return warnings.
- Failed primary table registration: continue with other files.
- Failed sidecar registration: keep primary table artifacts.

Tests may assert stricter database behavior by constructing
`BurstMFDBPipeline(db)` with an explicit temporary `MFDatabase`.

## Canonical RPC Surface

Only canonical dotted method names are part of the reference contract:

- `burst_selection.jobs.analyze_files`
- `burst_selection.results.inspect_bur`
- `burst_selection.gmm.fit`
- `burst_selection.diagnostics.load`
- `burst_selection.contract.describe`

Do not add legacy aliases for new plugins. If a compatibility alias is
temporarily required during migration, keep it outside the reference contract and
remove it before declaring the plugin a reference implementation.

## What Future Plugins Should Copy

Use Burst Selection as the template for new workflow-ready plugins:

1. Define request, settings, result, and optional archival context dataclasses in
   `api/models.py`.
2. Put payload normalization and canonical method names in `api/contract.py`.
3. Keep scientific code in pure `api/` functions.
4. Preserve per-input outputs when one run can produce repeated roles.
5. Put MFDB registration in plugin-owned `api/mfdb.py`.
6. Type plugin archival dependencies against `MFDBClientBase`.
7. Use `register_result()` and typed payload models instead of manual MFDB row
   writes.
8. Run MFDB archival after successful analysis in `backend/services.py`.
9. Keep GUI code as a request builder and result viewer.
10. Preflight GUI-driven MFDB output against the sample registry before starting
   irreversible processing.
11. Return warnings for optional archival failures instead of failing analysis.
12. Add focused MFDB tests with a real temporary `MFDatabase`.

## Verification Commands

Run the reference implementation checks:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
  /Users/tpeulen/mambaforge/envs/arm64/bin/python3 \
  -m pytest -p no:cov -o addopts='' \
  test/fio/test_payload_codec.py \
  test/fio/test_result_registry.py \
  test/fio/test_burst_pipeline_mfdb.py

PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
  /Users/tpeulen/mambaforge/envs/arm64/bin/python3 \
  -m pytest -p no:cov -o addopts='' \
  chisurf/plugins/burst/burst_selection/tests

rg 'artifact_type|storage_mode="local"|status="success"|parameter_name|parameter_value|db\.con' \
  chisurf/plugins/burst/burst_selection \
  test/fio/test_burst_pipeline_mfdb.py
```
