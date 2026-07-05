---
type: Plugin Profile
title: Trace Browser plugin
description: OKF profile for the TTTR trace browser.
resource: chisurf/plugins/tttr/trace_browser/
tags: [plugins, tttr, traces, cli, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `trace_browser` |
| Display name | `Spectroscopy:Single-Molecule:Trace Browser` |
| Categories | `Spectroscopy`, `Single-Molecule` |
| Version | `2.0.0` |
| State namespace | `trace_browser` |
| Local README | Missing |

The manifest describes folder browsing for PTU/TTTR intensity traces, file rating
and annotation, trace preview, and selected-trace export.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| API | `api/contract.py`, `api/io.py`, `api/models.py`. |
| Core | `core/metadata.py`, `core/trace.py`. |
| Backend services | `backend/services.py`. |
| CLI | `cli/main.py`, manifest command `trace-browser=...`. |
| GUI | `gui/tool.py`, `gui/client.py`. |
| Tests | `test/test_api.py`, `test_manifest.py`, `test_widgets.py`. |

Manifest RPC methods include file listing, metadata get/set, trace loading, CSV
export, and contract description.

# Data And Provenance Impact

The plugin works on photon-stream folders and stores ratings or annotations. The
README must explain where metadata is persisted, whether original TTTR files are
modified, and what the exported CSV contains.

# Verification Surface

Focused test command:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/tttr/trace_browser/test
```

# Documentation Work

- Add `chisurf/plugins/tttr/trace_browser/README.md`.
- Document accepted input folders/files and metadata sidecar behavior.
- Add CLI examples for list/export workflows.
- Link API contract details to `api/contract.py`.
