---
type: Subsystem
title: Project Persistence
description: The GUI-independent `.csp` project archive format and helpers for fit, dependency, and UI state.
resource: chisurf/core/project/
tags: [core, project, persistence, archive]
timestamp: '2026-07-05T00:00:00Z'
---

# Project archive

`chisurf/core/project/` provides the GUI-independent project format. A project
is saved as a `.csp` ZIP archive containing at least `project.json`; optional
entries include history, a chinet session, embedded data, and MFDB export
material.

| Module | Role |
| --- | --- |
| `project.py` | `Project` dataclass, JSON serialization, `.csp` save/load. |
| `archive.py` | `ProjectArchive`, safe in-memory ZIP handling and archive constants. |
| `fit_state.py` | Fit serialization helpers. |
| `ui_state.py` | Best-effort Qt window/dock/MDI/history-browser state capture and restore. |
| `registry.py` | Project metadata/registry support. |

# Current format

`Project.project_format_version` is `4`. The format is UID-keyed and rejects
pre-v4 projects at load time. Core sections are `datasets`, `experiments`,
`fits`, `dependency_edges`, `parameters`, `ui`, `extra`, and metadata. Dataset
and fit lookups are by UID, matching the migration away from Python object
identity in the [API facade](/architecture/api-facade.md) and [server DTOs](/architecture/server.md).

# Runtime use

Project save/load is exposed through macros, the [action layer](/architecture/action-layer.md),
and the server `project.*` RPC namespace. GUI-only state is deliberately kept
best-effort: failure to restore geometry or a dock layout should not make the
scientific project unreadable.

See also [data model](/subsystems/data-model.md), [operation history](/subsystems/history.md),
and the [MFDB store](/architecture/mfdb.md).
