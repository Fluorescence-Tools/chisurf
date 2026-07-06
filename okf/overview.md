---
type: Codebase
title: ChiSurf
description: Interactive global-analysis platform for time-resolved and single-molecule fluorescence data (TCSPC, FCS, smFRET).
resource: https://github.com/fluorescence-tools/chisurf
tags: [fluorescence, tcspc, fcs, smfret, python, qt]
timestamp: '2026-07-05T00:00:00Z'
---

# What it is

ChiSurf is an interactive global-analysis platform for time-resolved and
single-molecule fluorescence data — time-correlated single-photon counting
(TCSPC), fluorescence correlation spectroscopy (FCS), and single-molecule
FRET (smFRET). It ships a Qt GUI, a headless server, a CLI, and an
extensible plugin system.

# Source layout

| Path | Role |
|------|------|
| `chisurf/__init__.py` | Runtime globals, lazy accessors, logging, compatibility shims |
| `chisurf/__main__.py` | GUI entry point for `python -m chisurf` |
| `chisurf/core/` | Domain objects, fitting, data, models, math, settings, actions, API facade |
| `chisurf/gui/` | Qt application, widgets, plots, AutoForm renderer, GUI startup helpers |
| `chisurf/server/` | Headless, Qt-free ZMQ/JSON-RPC server |
| `chisurf/plugins/` | Built-in plugin packages, discovered via `manifest.json` |
| `chisurf/history/` | Operation-history recording and replay |
| `chisurf/macros/` | Scriptable convenience entry points for GUI, console, plugins |
| `modules/` | Compiled C++ extensions (chinet, ndxplorer, clsmview, quest) and the vendored `mfdb` package |
| `okf/prds/` | Numbered PRD design notes (`prd-NN.md`) driving current work |
| `docs/architecture.md` | Maintained source-of-truth for architecture |

# Entry points

- `python -m chisurf` (`pixi run chisurf`) — the GUI application.
- `csc` — the CLI.
- `python -m chisurf.server` — the headless ZMQ/JSON-RPC server.
- `csg_*` — per-plugin GUI scripts (see `[project.gui-scripts]`).

# Where to go next

- The [hybrid architecture](/architecture/index.md) — how state and calls
  flow through the API facade, action layer, and server.
- The [subsystems](/subsystems/index.md) — the major code areas.
- The [developer workflows](/workflows/index.md) — build and test.

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)
