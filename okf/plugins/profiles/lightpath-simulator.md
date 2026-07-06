---
type: Plugin Profile
title: Light Path Simulator plugin
description: OKF profile for the optical light-path simulator.
resource: chisurf/plugins/core/lightpath_simulator/
tags: [plugins, spectroscopy, rpc, cli]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `lightpath_simulator` |
| Display name | `Spectroscopy:Light Path Simulator` |
| Categories | `Spectroscopy` |
| Version | `1.0.0` |
| State namespace | `lightpath_simulator` |
| Local README | Missing |

The manifest describes an optical light-path simulator for crosstalk and R0 overlap
integral calculations.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| API | `api/client.py`, `api/contract.py`. |
| Core | `core/session.py`, `core/workflow.py`. |
| Backend | `backend/crosstalk.py`, `backend/simulator.py`, `backend/mmcif_export.py`. |
| RPC | `rpc/services.py`, `rpc/methods.py`, `rpc/client.py`. |
| CLI | `cli/main.py`, plus manifest command `lightpath-simulator=...`. |
| GUI | `gui/tool.py`, `gui/easy_mode.py`, `gui/node_types.py`. |
| Templates | Several JSON optical-path templates. |
| Tests | `tests/test_crosstalk.py`, `test_easy_mode.py`, `test_headless.py`. |

Manifest RPC methods include `lightpath.simulate`, `save`, `list`, `get`,
`get_probes_info`, and `contract.describe`.

# Data And Provenance Impact

The plugin is a strong reference candidate for the client-server plugin standard:
it has explicit API, core, backend, RPC, CLI, GUI, templates, and tests. It should
document whether saved sessions are local plugin state, MFDB records, or ordinary
files.

# Verification Surface

Focused test command:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/core/lightpath_simulator/tests
```

# Known Gotchas

- **Two preset formats coexist.** The Full Simulator saves optical-path presets
  as a node/edge **graph dict** (`{"nodes":…, "edges":…}`) under
  `~/.chisurf/presets/lightpath_optical/`, while Easy Mode's `_populate_form()`
  expects a **simplified config dict** (`lasers`/`detectors`/
  `emission_splitters`). Loaders must detect `"nodes" in cfg` and convert via
  `_graph_to_config()` before populating the Easy Mode form, or detectors/
  splitters silently collapse to one.
- **Graph loads need normalization.** Route Full Simulator graph loads through
  `normalize_lightpath_graph()` — it repairs missing splitter→bandpass→detector
  edges and remaps legacy output-relative source ports to global port indices;
  otherwise nodes render disconnected after a preset load.
- **Detector count.** Compute displayed detectors as
  `max(len(splitters)+1, len(detectors))`, and infer a splitter when detectors
  ≥ 2 but none are present, so multi-detector presets are not truncated.
- **macOS combobox opacity.** Node-embedded `QComboBox` popups render translucent
  on macOS; give them an opaque palette + `setAutoFillBackground(True)` and clear
  `WA_TranslucentBackground` on the view (a shared pitfall — see
  [Known issues & gotchas](/references/known-issues.md)).

# Documentation Work

- Add `chisurf/plugins/core/lightpath_simulator/README.md`.
- Add `docs/CONTRACT.md` if `api/contract.py` is expected to be stable.
- Document template file format and CLI examples.
- State persistence location and MFDB involvement explicitly.
