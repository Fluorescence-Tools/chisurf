# Burst Selection Workflow Contract

Burst Selection is the reference ChiSurf plugin for workflow-ready plugins. It
has a defined input model, output model, API layer, CLI adapter, ZMQ/RPC adapter,
and GUI client.

## Boundary Rule

Keep responsibilities split:

- `api/`: pure Python domain API. No Qt, no RPC transport, no CLI parsing.
- `api/models.py`: dataclass input/output models.
- `api/contract.py`: JSON-safe workflow contract, RPC method names, and payload normalization.
- `backend/services.py`: ServiceDispatcher/RPC handlers. It accepts JSON payloads and delegates to `api/`.
- `server/`: compatibility ZMQ entry points. It delegates to `backend/services.py`.
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
  "selected_setup": "Test"
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
    "metadata": {
      "n_files": 1,
      "n_bursts": 0,
      "n_selected": 0,
      "n_photons": 0
    }
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

Legacy aliases remain available for migration:

- `burst_selection.analyze_files`
- `burst_selection.inspect_bur`
- `burst_selection.fit_gmm_from_bur`

New integrations should use the canonical dotted names.
