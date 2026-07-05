---
type: Subsystem
title: Macros, CLI & Scripting
description: How ChiSurf is driven from code — recordable macros, the `csc` Click CLI, and the recording QtConsole.
resource: chisurf/macros/
tags: [scripting, core, cli]
timestamp: '2026-07-05T00:00:00Z'
---

# Scope

Three overlapping ways to drive ChiSurf without clicking the GUI:

| Surface | Entry | Code |
| --- | --- | --- |
| Macros | `import chisurf.macros` | `chisurf/macros/` |
| CLI | `csc <sub> …` | `chisurf/core/cli.py` |
| QtConsole | in-GUI IPython dock | `chisurf/gui/widgets/ipython.py` |

All three ultimately go through the [API facade](/architecture/api-facade.md)
(`chisurf.core.api`, the stable surface for "GUI, macros, plugins, and
QtConsole") and record into [history](/subsystems/history.md).

# Macros (`chisurf/macros/`)

Plain Python functions that perform domain operations; the package re-exports
`core_fit` and `core_data` with `*` and pulls in `model`, `model_parse`,
`plugin_check`.

| Module | Role |
| --- | --- |
| `core_data.py` | `add_dataset`, `group_datasets`, `remove_datasets`, `reinitialize_application`, reader resolution |
| `core_fit.py` | fit lifecycle, linking, project archive save/load, `export_action_catalog` |
| `model.py`, `model_parse.py` | model configuration / parsed-model control |
| `plugin_check.py` | `PluginTestRunner` — plugin self-test logic (UI-free) |

Macros call `record_action(...)` (`chisurf/macros/*._record_history`) so
scripted state changes land in the [action layer](/architecture/action-layer.md)
and history exactly like GUI clicks.

# CLI — `csc` (`chisurf/core/cli.py`)

A Click `Group` (`PluginCLI`) that lazily discovers and registers
plugin-provided subcommands on first invocation (no plugin import at load
time). Declared as the `csc` console script in `[project.scripts]`; a second
script `chimol-cli` targets the chimol app. Examples from the group docstring:
`csc lltf`, `csc burst-background`, `csc count-rate analyze`.
[PRD-30](/prds/prd-30.md) extends these into stdin/stdout Unix pipe stages.

# GUI entry points (`[project.gui-scripts]`)

`chisurf` (`chisurf.__main__:main`) plus ~20 standalone `csg_*` plugin GUIs
(e.g. `csg_kappa2distribution`, `csg_fret_dock`, `csg_ndxplorer`,
`csg_batch_analysis`) and `chisurf_update`. Each maps to a plugin `__main__:main`.
The headless [server](/architecture/server.md) runs via `python -m chisurf.server`.

# QtConsole (`chisurf/gui/widgets/ipython.py`)

`QIPythonWidget(RichJupyterWidget)` embeds an in-process Jupyter kernel with
ChiSurf's namespace preloaded. It supports **macro recording**: `start_recording`
/ `stop_recording` accumulate executed code into `self._macro`, which
`save_macro`/`run_macro` write to and replay from `.py` files — the manual
counterpart to history-based replay.

# Scriptability & tests

Because the [core](/subsystems/core.md) is Qt-free, macros and `csc` run
headless against `ChiSurfAPI` in `local`, `hybrid`, or `server` mode.
[PRD-46](/prds/prd-46.md) promotes such scripts to first-class citizens of the
test pipeline.

See also: [overview](/overview.md), [data IO](/subsystems/data-io.md),
[compiled modules](/subsystems/compiled-modules.md).
