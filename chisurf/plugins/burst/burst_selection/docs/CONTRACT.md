# Burst Selection Workflow Contract

Burst Selection is the reference ChiSurf plugin for workflow-ready plugins. It
has a defined input model, output model, API layer, CLI adapter, ZMQ/RPC adapter,
GUI client, and MFDB archival adapter. The broader implementation pattern is
documented in `REFERENCE_IMPLEMENTATION.md`.

## Boundary Rule

Keep responsibilities split:

- `api/`: pure Python domain API. No Qt, no RPC transport, no CLI parsing.
- `api/models.py`: dataclass input/output models.
- `api/contract.py`: JSON-safe workflow contract, RPC method names, and payload normalization.
- `api/mfdb.py`: plugin-owned MFDB archival for successful analysis results.
- `backend/services.py`: ServiceDispatcher/RPC handlers. It accepts JSON payloads and delegates to `api/`.
- `server/`: ZMQ entry points. It delegates to `backend/services.py`.
- `cli/`: command-line adapter. It builds the same `AnalysisRequest` used by RPC.
- `gui/`: presentation only. It calls `BurstSelectionClient`; it should not own analysis behavior.

Future workflow/node systems should call the API or RPC contract. They should
not depend on PyQt widgets or legacy wizard state.

## Discovering The Contract

CLI:

```bash
chisurf burst-selection contract
```

RPC:

```json
{
  "method": "burst_selection.contract.describe",
  "params": {}
}
```

Python:

```python
from chisurf.plugins.burst.burst_selection.api import contract_descriptor

contract = contract_descriptor()
```

## Main Analysis Input

Canonical RPC method:

```text
burst_selection.jobs.analyze_files
```

Input payload:

```json
{
  "files": ["/data/m000.spc", "/data/m001.spc"],
  "filetype": null,
  "windows": {
    "prompt": [0, 2048],
    "delayed": [2048, 4095]
  },
  "detectors": {
    "green": {
      "chs": [8, 0, 3],
      "micro_time_ranges": [[0, 4095]],
      "g_factor": 1,
      "l1": 0,
      "l2": 0
    }
  },
  "settings": {
    "photon_filter": {
      "channels": [0, 1, 8, 9],
      "microtime_ranges": [],
      "filter_active": true,
      "used_filter": "burst"
    },
    "burst_detection": {
      "min_photons": 60,
      "photon_window": 5,
      "time_window": 0.06
    },
    "output_formats": ["bur"],
    "zip_output": false,
    "remove_folder": false
  },
  "output_dir": "/tmp/output",
  "legacy_output": true,
  "selected_setup": "Test",
  "mfdb": {
    "enabled": true,
    "sample_id": "sample_1",
    "source_artifact_ids": {},
    "register_missing_inputs": true,
    "setup_id": "tttr_detector_setup:bh_spc_130",
    "setup_version": 1
  }
}
```

The adapter normalizes this payload to `AnalysisRequest`.

## Main Analysis Output

The service envelope is always:

```json
{
  "ok": true,
  "result": {
    "files": ["/data/m000.spc"],
    "dataframes": {
      "/data/m000.spc": []
    },
    "feature_dataframe": null,
    "gmm_fit": null,
    "output_paths": {
      "bur": "/tmp/output/m000.bur",
      "output_folder": "/tmp/burstwise_All 0.2000#60"
    },
    "output_paths_by_file": {
      "/data/m000.spc": {
        "bur": "/tmp/output/m000.bur"
      }
    },
    "metadata": {
      "n_files": 1,
      "n_bursts": 0,
      "n_selected": 0,
      "n_photons": 0
    },
    "mfdb_artifacts": {
      "input_artifacts": {
        "/data/m000.spc": "raw_artifact_id"
      },
      "burst_table_artifacts": {
        "/data/m000.spc": "burst_table_artifact_id"
      },
      "sidecar_artifacts": {},
      "warnings": []
    },
    "warnings": []
  }
}
```

On failure:

```json
{
  "ok": false,
  "error": "message",
  "error_code": "operation_failed"
}
```

## Supported Methods

- `burst_selection.jobs.analyze_files`: process TTTR files and write configured outputs.
- `burst_selection.results.inspect_bur`: inspect a saved `.bur` file.
- `burst_selection.gmm.fit`: fit a GMM to `.bur` features.
- `burst_selection.diagnostics.load`: JSON-safe diagnostics for non-GUI clients.
- `burst_selection.contract.describe`: return the machine-readable contract descriptor.

No legacy aliases are part of the reference contract. New integrations should
use only the canonical dotted names.

## MFDB Context

MFDB archival context is nested under `mfdb`:

```json
{
  "mfdb": {
    "enabled": true,
    "sample_id": "sample_1",
    "source_artifact_ids": {
      "/data/m000.spc": "existing_raw_artifact_id"
    },
    "register_missing_inputs": true,
    "setup_id": "tttr_detector_setup:bh_spc_130",
    "setup_version": 1
  }
}
```

Semantics:

- `enabled=false`: skip archival.
- `sample_id`: link registered artifacts to an existing sample.
- `source_artifact_ids`: reuse pre-existing raw artifacts by normalized input path.
- `register_missing_inputs=true`: archive inputs not already present in
  `source_artifact_ids`.
- `setup_id`: link MFDB operations to the detector/PIE setup in `mfdb_setup`.
- `setup_version`: optional setup version stored in operation/artifact metadata.

GUI behavior:

- MFDB is an explicit output mode and may be selected without CSV/HDF5 output.
- When MFDB output is selected, the GUI checks whether each raw input's content
  MD5 is already associated with a sample in the object store.
- If the raw content has no sample association, the GUI opens the sample
  registration dialog before calling `burst_selection.jobs.analyze_files`.
- The resulting request contains the selected `sample_id` and any pre-registered
  raw artifacts under the nested `mfdb` object shown above.
- The selected detector setup is stored in MFDB before analysis and included as
  `mfdb.setup_id`, so processed data can be traced to the setup used.

Top-level provenance fields are intentionally not part of the contract.
