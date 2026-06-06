# ChiSurf Software Architecture

This document describes the current source layout and runtime architecture of
ChiSurf. It is intentionally implementation-oriented: if this file disagrees
with `chisurf/`, the source tree wins and this document should be fixed.

## Source Layout

| Path | Role |
|------|------|
| `chisurf/__init__.py` | Runtime globals, lazy accessors, logging setup, compatibility shims |
| `chisurf/__main__.py` | GUI application entry point for `python -m chisurf` |
| `chisurf/core/` | Domain objects, fitting, data, models, math, settings, actions, API facade |
| `chisurf/core/actions/` | Action registry, dispatcher, and state-change action implementations |
| `chisurf/core/api/` | Hybrid API facade, plugin context, remote client wrapper, optional proxies |
| `chisurf/gui/` | Qt application, widgets, plots, resources, and GUI startup helpers |
| `chisurf/history/` | Operation-history recording and replay support |
| `chisurf/macros/` | Scriptable convenience entry points used by GUI, console, and plugins |
| `chisurf/plugins/` | Built-in plugin packages and plugin utilities |
| `chisurf/server/` | Headless ZMQ/JSON-RPC server, session state, dispatcher, services, transport |

## Runtime Layers

```text
Presentation
  chisurf.gui, chisurf.plugins, chisurf.macros, scripts/CLI
      |
      v
Facade and Action Routing
  chisurf.core.api.ChiSurfAPI
  chisurf.core.api.PluginContext
  chisurf.core.actions.ActionDispatcher / ActionRegistry
      |
      v
Service and Transport
  chisurf.server.app.ChiSurfServer
  chisurf.server.dispatcher.ServiceDispatcher
  chisurf.server.transport.zmq.ZmqServer / ZmqClient
      |
      v
Domain and Session State
  chisurf.core data/model/fitting objects
  chisurf.server.session.SessionState
  runtime globals: chisurf.fits, chisurf.imported_datasets, chisurf.cs
```

## Important Runtime Globals

`chisurf/__init__.py` still exposes several process-local globals. These are
part of the current hybrid architecture and remain important for GUI and legacy
macro compatibility.

| Global | Meaning |
|--------|---------|
| `chisurf.fits` | Process-local list of current fit groups |
| `chisurf.imported_datasets` | Process-local list of imported datasets |
| `chisurf.cs` | Current Qt main window instance in the GUI process |
| `chisurf.experiment` | Registered experiment objects keyed by name |
| `chisurf.working_path` | Current working path used by GUI and macros |
| `chisurf.action_dispatcher` | Lazily created `ActionDispatcher` |
| `chisurf.action_registry` | Dispatcher registry for action metadata |
| `chisurf.action_catalog` | Callable returning action catalogue metadata |
| `chisurf.action_execute` | Callable for action execution by canonical or dotted name |

These globals are not the target architecture for server-owned state. New code
that needs datasets, fits, parameters, project state, or server communication
should prefer `chisurf.core.api.ChiSurfAPI` or `chisurf.core.api.PluginContext`.

## Action Layer

The action layer lives under `chisurf/core/actions/`.

| File | Role |
|------|------|
| `_infra.py` | `ActionSpec`, `ActionRegistry`, `ActionDispatcher`, default dispatcher helpers |
| `_decorator.py` | Action registration decorator support |
| `dataset_actions.py` | Dataset state-change actions |
| `fit_actions.py` | Fit state-change actions |
| `model_actions.py` | Model actions |
| `parameter_actions.py` | Parameter actions |
| `project_actions.py` | Project actions |

`chisurf.__getattr__` lazily exposes `action_dispatcher`, `action_registry`,
`action_catalog`, and `action_execute`. Action names may be canonical internal
names or dotted aliases, depending on the registered action spec.

## API Facade

`chisurf.core.api.ChiSurfAPI` is the stable facade for GUI, macros, plugins, and
the QtConsole.

Modes:

| Mode | Behavior |
|------|----------|
| `local` | Read and mutate current in-process objects for legacy workflows |
| `hybrid` | Preserve local behavior while allowing migrated server paths |
| `server` | Route operations through `ChisurfClient` RPC calls |

Related classes and modules:

| Symbol | Path | Role |
|--------|------|------|
| `ChiSurfAPI` | `chisurf.core.api` | Dataset, fit, parameter, project, session facade |
| `PluginContext` | `chisurf.core.api.context` | API/client/main-window context passed to migrated plugins |
| `ChisurfClient` | `chisurf.core.api._client` | High-level client over ZMQ JSON-RPC transport |
| `RemoteError` | `chisurf.core.api._client` | Client-side structured RPC error exception |

## Server Architecture

ChiSurf includes a headless server based on ZMQ and JSON-RPC 2.0. The server is
Qt-free and lives under `chisurf/server/`.

| Path | Role |
|------|------|
| `app.py` | `ChiSurfServer`, server wiring and lifecycle |
| `__main__.py` | `python -m chisurf.server` entry point |
| `startup.py` | Subprocess startup/termination helpers |
| `dispatcher.py` | `ServiceDispatcher`, method registration and invocation |
| `server_methods.json` | Declarative server RPC registry |
| `client_methods.json` | Declarative client wrapper method registry |
| `protocol.py` | JSON-RPC encode/decode helpers and protocol metadata |
| `session.py` | `SessionState`, server-side runtime state container |
| `dto.py` | Dataclasses documenting JSON-safe DTO contract shapes |
| `eventbus.py` | In-process event bus feeding ZMQ PUB/SUB events |
| `jobs.py` | Job lifecycle helpers for long-running work |
| `services/` | Dataset, fit, parameter, project, session, model, and graph handlers |
| `transport/zmq.py` | ZMQ REP/PUB server and REQ/SUB client transport |

Services receive a `SessionState` and return `ServiceResult` dictionaries with
`{"ok": bool, ...}`. Failures should use `chisurf.server.services.service_error()`
so callers can inspect `error_code`, `jsonrpc_code`, and optional
`exception_type` fields.

## RPC Namespaces

The authoritative method registry is `chisurf/server/server_methods.json`.
`meta.protocol` returns the namespace catalogue from `chisurf.server.protocol`.

| Namespace | Methods |
|-----------|---------|
| `meta` | `meta.ping`, `meta.methods`, `meta.protocol` |
| `dataset` | `dataset.list`, `dataset.get`, `dataset.curve_data`, `dataset.load`, `dataset.rename`, `dataset.group`, `dataset.ungroup`, `dataset.remove`, `dataset.clear` |
| `fit` | `fit.list`, `fit.get`, `fit.create`, `fit.run`, `fit.update`, `fit.save`, `fit.curve_data`, `fit.set_dataset`, `fit.set_result_idx`, `fit.set_fit_range`, `fit.remove`, `fit.clear` |
| `parameter` | `parameter.get`, `parameter.set_value`, `parameter.set_fixed`, `parameter.set_bounds`, `parameter.set_bounds_on`, `parameter.link`, `parameter.unlink` |
| `project` | `project.info`, `project.save`, `project.load` |
| `session` | `session.describe`, `session.clear`, `session.snapshot`, `session.restore` |
| `model` | `model.finalize`, `model.set_parse_function` |
| `graph` | `graph.build`, `graph.build_fits` |

Legacy aliases such as `list_datasets`, `get_fit_info`, `run_fit`, and
`save_project` remain registered for compatibility while code migrates.

## DTO Policy

`chisurf/server/dto.py` contains dataclasses such as `DatasetSummary`,
`FitSummary`, `FitDetail`, `ParameterDTO`, `SetupDTO`, `ProjectInfoDTO`, and
`ActionResultDTO`. These classes document the JSON contract and provide helper
serialization. Service handlers are not required to return dataclass instances;
they should return JSON-safe dictionaries matching the documented shapes.

DTOs must not contain Qt objects or arbitrary Python domain objects. Stable IDs
should be exposed as `uid` strings where possible. GUI code should compare DTOs
by `uid`, not by Python object identity.

## Plugin System

Built-in plugins live under `chisurf/plugins/`. User plugins live under
`~/.chisurf/plugins/`.

A plugin package normally contains an `__init__.py` with a `name` variable:

```python
name = "Category:Plugin Name"
```

The category before `:` controls the plugin-menu grouping. Plugin entry code is
usually guarded by `if __name__ == "plugin":` so regular imports do not launch
widgets.

Migrated plugins should use `PluginContext` or `ChiSurfAPI` for datasets, fits,
parameters, project state, and server-owned state. They may continue to use Qt
objects locally for UI work.

## Project Persistence

Project save/load is implemented in the core project/fitting code and exposed
through `project.save`, `project.load`, and `project.info` RPC methods. Project
files bundle fit configurations, model parameters, data references, history
metadata, and plugin/action catalogue extras where supported.

## Design Constraints

| Constraint | Rationale |
|------------|-----------|
| `chisurf.server` must not import Qt or `chisurf.gui` | Server must run headless and in subprocesses |
| Use ZMQ plus JSON-RPC 2.0 only for server communication | Keeps one transport/protocol contract |
| Do not install transparent object proxies in normal GUI startup | The GUI still expects real Python objects in many paths |
| Use explicit DTOs and commands for server-owned state | JSON cannot preserve Python identity, `isinstance`, or deep mutation semantics |
| Prefer additive migration steps | The GUI and plugins must keep working during the hybrid period |
| Keep documentation source-aligned | Wrong docs are worse than missing docs |
