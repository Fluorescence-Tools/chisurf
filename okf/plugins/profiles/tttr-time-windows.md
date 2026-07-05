---
type: Plugin Profile
title: TTTR Time Windows plugin
description: OKF profile for TTTR-to-time-window BID generation.
resource: chisurf/plugins/tttr/tttr_time_windows/
tags: [plugins, tttr, converter, cli, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `tttr_time_windows` |
| Display name | `Tools:Converter:TTTR->Time-Window BIDs` |
| Categories | `Tools`, `Converter` |
| Version | `2.0.0` |
| State namespace | `tttr_time_windows` |
| Local README | Missing |

The manifest describes splitting TTTR files into fixed-duration time-window BID
`.bst` files.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| API | `api/contract.py`, `api/io.py`, `api/models.py`, `api/selection.py`, `api/serialization.py`. |
| Backend services | `backend/services.py`, `backend/state.py`. |
| Server compatibility | `server/services.py`, `server/methods.py`, `server/client.py`. |
| CLI | `cli/main.py`, manifest command `tttr-time-windows=...`. |
| GUI | `gui/tool.py`, `gui/client.py`, `gui/adapter.py`. |
| Tests | `tests/test_construction_smoke.py`. |

Manifest RPC methods include `tttr_time_windows.jobs.analyze_files` and
`tttr_time_windows.contract.describe`.

# Data And Provenance Impact

This plugin generates derived `.bst` files from TTTR sources. Docs should spell out
window duration, output directory rules, naming, overwrite behavior, and whether any
source-file provenance is embedded in the BID output.

# Verification Surface

Focused smoke test:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/tttr/tttr_time_windows/tests
```

# Documentation Work

- Add `chisurf/plugins/tttr/tttr_time_windows/README.md`.
- Add `docs/CONTRACT.md` if the API/RPC contract is meant to be stable.
- Document file naming, batch behavior, and downstream burst-selection assumptions.
- Clarify why both `backend/` and `server/` compatibility modules exist.
