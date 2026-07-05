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

# Documentation Work

- Add `chisurf/plugins/core/lightpath_simulator/README.md`.
- Add `docs/CONTRACT.md` if `api/contract.py` is expected to be stable.
- Document template file format and CLI examples.
- State persistence location and MFDB involvement explicitly.
