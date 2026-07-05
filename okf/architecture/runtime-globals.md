---
type: Architecture
title: Runtime Globals (legacy)
description: Process-local globals in chisurf/__init__.py and the migration away from them.
resource: chisurf/__init__.py
tags: [globals, legacy, migration, state]
timestamp: '2026-07-05T00:00:00Z'
---

# Status

`chisurf/__init__.py` still exposes several process-local globals. They are
part of the current hybrid architecture and remain important for GUI and
legacy macro compatibility, but they are **not** the target architecture for
server-owned state.

| Global | Meaning |
|--------|---------|
| `chisurf.fits` | Process-local list of current fit groups |
| `chisurf.imported_datasets` | Process-local list of imported datasets |
| `chisurf.cs` | Current Qt main-window instance in the GUI process |
| `chisurf.experiment` | Registered experiment objects keyed by name |
| `chisurf.working_path` | Current working path used by GUI and macros |
| `chisurf.action_dispatcher` | Lazily created `ActionDispatcher` |
| `chisurf.action_registry` | Dispatcher registry for action metadata |
| `chisurf.action_catalog` | Callable returning action-catalogue metadata |
| `chisurf.action_execute` | Callable for action execution by canonical or dotted name |

# Migration guidance

New code that needs datasets, fits, parameters, project state, or server
communication should prefer the [API facade](/architecture/api-facade.md)
(`ChiSurfAPI`) or `PluginContext`, not these globals.

# Citations

[1] [ChiSurf architecture doc](/references/architecture-doc.md)
