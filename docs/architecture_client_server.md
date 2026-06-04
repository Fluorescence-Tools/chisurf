# ChiSurf Client-Server Architecture

This document is the central reference for migrating ChiSurf to a clean
client-server architecture. It describes the target architecture, the current
hybrid state, and the constraints that must be respected while keeping the
application working.

Detailed implementation steps are in
[`docs/client_server_migration_plan.md`](client_server_migration_plan.md).

## Goals

- Move computation and core runtime state out of the Qt GUI process.
- Use **only ZMQ + JSON-RPC 2.0** for process communication.
- Keep the GUI responsive and mostly presentation-only.
- Keep plugins functional during migration and make their server interaction explicit.
- Allow multiple ChiSurf GUI instances without port or state collisions.
- Keep the application working at every migration step.

## Non-Goals

- Do not reintroduce FastMCP, HTTP, SSE, or other server protocols.
- Do not implement a fake distributed Python-object layer for GUI use.
- Do not require Qt in the server process.
- Do not force all plugins to migrate in one large change.

## Current State

The current codebase is hybrid:

- `chisurf.server` is a clean ZMQ/JSON-RPC server package.
- `ChiSurfServer` owns a `SessionState` in the server process.
- `ChisurfClient` can communicate with the server.
- The GUI starts a private server subprocess on dynamic ports.
- The GUI still computes locally and uses real in-process objects:
  - `chisurf.fits`
  - `chisurf.imported_datasets`
  - `chisurf.cs`
- Plugins and macros still directly access local globals and local objects.
- The proxy package exists for headless experiments but must **not** be installed by the GUI by default.

This state is intentionally stable: the GUI continues to work while the server
API is expanded and migrated into.

## Target Architecture

```text
┌──────────────────────────────┐       ZMQ/JSON-RPC       ┌──────────────────────────────┐
│ GUI Process                  │  ──────────────────────► │ Server Process                │
│                              │                          │                              │
│ Qt widgets                   │                          │ SessionState                  │
│ Plugin UI                    │                          │ Dataset objects               │
│ Plot widgets                 │                          │ Fit/model objects             │
│ QtConsole client facade      │                          │ Parameter objects             │
│                              │                          │ Project/session persistence   │
│ No heavy fitting             │ ◄──────────────────────  │ Compute services              │
│ No direct model computation  │      ZMQ PUB/SUB events  │ ServiceDispatcher             │
└──────────────────────────────┘                          └──────────────────────────────┘
```

### Server Responsibilities

- Own authoritative state:
  - datasets
  - fits
  - models
  - parameters
  - project/session data
- Execute computation:
  - file reading / dataset loading
  - fit creation
  - fit execution
  - model update/finalize
  - parameter updates and linking
  - project load/save
- Expose explicit JSON-safe DTOs.
- Emit events through ZMQ PUB/SUB.
- Never import `chisurf.gui`, `qtpy`, `PyQt5`, or widget code.

### GUI Responsibilities

- Own Qt widgets and view state only.
- Display DTO snapshots from the server.
- Send commands through `ChisurfClient` or a higher-level API facade.
- Subscribe to server events and refresh affected views.
- Keep `chisurf.cs` as the local Qt main window object.
- Avoid local heavy computation after migration of each workflow.

### Plugin Responsibilities

- Use a provided `PluginContext` or `chisurf.api` facade.
- Avoid direct mutation of:
  - `chisurf.fits`
  - `chisurf.imported_datasets`
  - server-owned parameters/models/datasets
- Avoid `chisurf.run("...")` for server-owned state changes.
- Keep Qt UI work local.

## Why Not Use Python Object Proxies For The GUI?

The existing GUI and plugins expect real Python objects. They use:

- `fit.model.parameters_all_dict`
- `fit.data.name`
- `fit.run()`
- `fit.update()`
- `isinstance(dataset, DataCurve)`
- `f is current_fit`
- `id(dataset)` comparisons
- direct assignment such as `p.link = parameter`

A JSON-backed proxy cannot correctly preserve Python identity, `isinstance`,
method dispatch, cyclic object graphs, or deep mutation tracking. Therefore the
target architecture is **explicit DTOs and commands**, not transparent object
proxying.

## API Naming Convention

New methods should be namespaced. Existing legacy aliases may remain while code
migrates.

| Domain | Methods |
|--------|---------|
| Session | `session.describe`, `session.clear`, `session.snapshot` |
| Dataset | `dataset.list`, `dataset.get`, `dataset.load`, `dataset.remove`, `dataset.clear` |
| Fit | `fit.list`, `fit.get`, `fit.create`, `fit.run`, `fit.update`, `fit.remove`, `fit.clear` |
| Parameter | `parameter.get`, `parameter.set_value`, `parameter.set_fixed`, `parameter.set_bounds`, `parameter.link`, `parameter.unlink` |
| Setup | `setup.list`, `setup.get`, `setup.set_property`, `setup.apply` |
| Project | `project.info`, `project.save`, `project.load` |
| Action | `action.execute`, `action.list` |
| Meta | `meta.ping`, `meta.methods` |

## DTO Principles

- DTOs must be JSON-serializable.
- DTOs must include stable identifiers (`uid`) wherever possible.
- GUI must compare by `uid`, not Python identity.
- DTOs should be explicit and versionable.
- DTOs should not contain Qt objects or arbitrary Python objects.

Example `FitSummary`:

```json
{
  "uid": "fit-uuid",
  "index": 0,
  "name": "Fit 1",
  "type": "FitGroup",
  "chi2": 1.23,
  "dataset_uid": "dataset-uuid",
  "dataset_name": "sample.ptu",
  "model_name": "LifetimeModel",
  "parameter_count": 12
}
```

Example `ParameterDTO`:

```json
{
  "name": "tau1",
  "value": 3.8,
  "fixed": false,
  "bounds": [0.0, 100.0],
  "bounds_on": true,
  "linked_to": null,
  "error_estimate": 0.1
}
```

## Event Model

The server publishes events on the PUB socket after state changes.

Required topics:

- `session.changed`
- `dataset.added`
- `dataset.removed`
- `dataset.cleared`
- `fit.added`
- `fit.updated`
- `fit.removed`
- `fit.cleared`
- `parameter.changed`
- `project.loaded`
- `project.saved`
- `job.started`
- `job.finished`
- `job.failed`

Event payloads must include enough identifiers for the GUI to refresh the
minimum required view.

## Compatibility Strategy

The application must work after every merge. Migration therefore proceeds in
small reversible steps:

1. Keep current GUI behavior intact.
2. Add server endpoint and tests.
3. Add client facade method.
4. Migrate one widget/plugin path to the facade.
5. Verify GUI still imports and server tests pass.
6. Remove old local-compute path only after all callers migrated.

## Current Critical Coupling Points

These areas must be migrated carefully:

- `chisurf/macros/core_data.py`
- `chisurf/macros/core_fit.py`
- `chisurf/macros/model.py`
- `chisurf/macros/model_parse.py`
- `chisurf/actions/*`
- `chisurf/gui/widgets/fitting/*`
- `chisurf/gui/widgets/experiments/*`
- Plugins using `chisurf.fits`, `chisurf.imported_datasets`, `chisurf.actions.dispatch`, or `chisurf.run`.

## Acceptance Criteria For Clean Architecture

The migration is complete when:

- The GUI can start without constructing core datasets/fits locally.
- Fit execution happens only through server RPC.
- Dataset loading happens only through server RPC.
- Project load/save happens only through server RPC.
- Plugins use `PluginContext` or `chisurf.api`, not direct globals, for server-owned state.
- `chisurf.fits` and `chisurf.imported_datasets` are no longer authoritative in the GUI process.
- The server package imports without Qt installed.
- Multiple GUI instances spawn independent server subprocesses.
- Server integration tests cover the main workflows.
