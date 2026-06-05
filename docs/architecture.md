# ChiSurf Software Architecture

This document describes the high-level architecture of ChiSurf, covering the
plugin system, client-server layer, project model, and data-flow patterns.

## System Overview

ChiSurf is organised in four main layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                    │
│  ┌──────────────┐  ┌────────────┐  ┌───────────────┐   │
│  │  Qt GUI       │  │  Plugins   │  │  Scripts/CLI  │   │
│  │  (gui/)       │  │  (plugins/)│  │  (macros/)    │   │
│  └──────┬───────┘  └─────┬──────┘  └───────┬───────┘   │
│         │                │                  │           │
│         └────────────────┼──────────────────┘           │
│                          │                              │
│  ┌───────────────────────┼──────────────────────┐       │
│  │           Action Layer (runtime/actions.py)   │       │
│  │  ActionController → Dispatcher → History      │       │
│  └───────────────────────┼──────────────────────┘       │
│                          │                              │
│  ┌───────────────────────┼──────────────────────┐       │
│  │           Service Layer                        │       │
│  │  ┌──────────┐ ┌──────┴───────┐ ┌──────────┐ │       │
│  │  │  Server   │ │  Dispatcher  │ │  Client   │ │       │
│  │  │  (app.py) │ │  (dispatch)  │ │(client.py)│ │       │
│  │  └────┬─────┘ └──────┬───────┘ └─────┬────┘ │       │
│  │       │              │                │        │       │
│  │       └──── ZMQ ─────┴──── JSON-RPC ──┘        │       │
│  └───────────────────────┼──────────────────────┘       │
│                          │                              │
│  ┌───────────────────────┼──────────────────────┐       │
│  │              Domain Layer                       │       │
│  │  chisurf.fits, chisurf.imported_datasets,       │       │
│  │  chisurf.project, chisurf.models, ...           │       │
│  │  EventBus, JobManager, SessionState             │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| **Presentation** | `chisurf.gui`, `chisurf.plugins`, `chisurf.macros` | User interaction, visualisation, scripting entry points |
| **Action** | `chisurf.actions._infra` | State-change routing, history recording, MCP bridge |
| **Service** | `chisurf.server`, `chisurf.client` | Remote procedure call layer, dispatcher, ZMQ transport |
| **Domain** | `chisurf.fits`, `chisurf.models`, `chisurf.project`, ... | Core data structures, algorithms, persistence |

---

## Plugin System

### Discovery

Plugins are loaded from two locations:

| Location | Purpose |
|----------|---------|
| `chisurf/plugins/` | Built-in plugins shipped with ChiSurf |
| `~/.chisurf/plugins/` | User-defined custom plugins (created automatically on startup) |

At startup ChiSurf scans both directories for Python packages. Each package
must contain an `__init__.py` that defines the plugin metadata and entry point.

### Plugin Structure

A minimal plugin directory:

```
my_plugin/
├── __init__.py      # Plugin code + metadata
├── create_icon.py   # Script to generate the icon
└── icon.png         # 48×48 menu icon
```

### Plugin Metadata

Every plugin must define a `name` variable in its `__init__.py`:

```python
name = "Category:Plugin Name"
```

The string before the colon (`:`) determines the submenu category in the
Plugins menu. Common categories:

| Category | Purpose |
|----------|---------|
| `Tools` | Utility tools |
| `Structure` | Structural biology / molecular dynamics |
| `Analysis` | Data analysis routines |
| `Visualization` | Data viewers and plots |
| `Processing` | Data transformation |
| `Import/Export` | File format converters |

### Plugin Entry Point

Plugin code is executed when the module is loaded with `__name__ == "plugin"`:

```python
if __name__ == "plugin":
    window = MyPluginWidget()
    window.show()
```

This guard ensures the plugin only activates when ChiSurf loads it via the
plugin manager, not during regular `import`.

### Plugin Manager

The Plugin Manager (Tools → Plugin Manager) lets users:

- **Enable/disable** individual plugins
- **View** plugin metadata (name, path, category, icon)
- **Distinguish** built-in vs. user plugins (user plugins show `[user]` indicator)

### Creating Plugins

**Cookiecutter (recommended):**

```bash
pip install cookiecutter
cookiecutter path/to/chisurf/plugins/cookiecutter-chisurf-plugin
```

**Manual:** Create a directory with `__init__.py` following the structure above,
then copy it to `~/.chisurf/plugins/`.

### Plugin Lifecycle

```
discovery ──► scan directories ──► load __init__.py ──►
    register in PluginManager ──► user enables ──►
    execute with __name__=="plugin"
```

---

## Client-Server Architecture

### Overview

ChiSurf includes a lightweight RPC server based on **ZeroMQ** and
**JSON-RPC 2.0**. The server enables programmatic control of ChiSurf from
external processes (scripts, other applications, MCP agents) without
requiring GUI access.

### Components

```
┌───────────────────────────────┐    ┌───────────────────────────────┐
│         Server Process        │    │        Client Process         │
│                               │    │                               │
│  ┌─────────────────────────┐  │    │  ┌─────────────────────────┐  │
│  │     ChiSurfServer        │  │    │  │     ChisurfClient       │  │
│  │  ┌───────────────────┐   │  │    │  │                         │  │
│  │  │ ServiceDispatcher  │   │  │    │  │  list_datasets()       │  │
│  │  │  dispatch(m,p) → r │   │  │    │  │  list_fits()           │  │
│  │  └────────┬──────────┘   │  │    │  │  run_fit()             │  │
│  │           │              │  │    │  │  set_parameter_value()  │  │
│  │  ┌────────┴──────────┐   │  │    │  │  ...                   │  │
│  │  │   Service Handlers │   │  │    │  └──────────┬────────────┘  │
│  │  │  datasets / fits   │   │  │    │             │               │
│  │  │  parameters / proj │   │  │    │  ┌──────────┴────────────┐  │
│  │  └────────┬──────────┘   │  │    │  │      ZmqClient        │  │
│  │           │              │  │    │  │  call(method, params)  │  │
│  │  ┌────────┴──────────┐   │  │    │  └──────────┬────────────┘  │
│  │  │      ZmqServer    │   │  │    │             │               │
│  │  │  REP socket : cmd  │   │  │    │     REQ socket             │
│  │  │  PUB socket : pub  │   │  │    │     SUB socket             │
│  │  └───────────────────┘   │  │    │  └─────────────────────────┘  │
│  └─────────────────────────┘  │    └───────────────────────────────┘
│                               │
│  ┌─────────────────────────┐  │
│  │   Global State          │  │
│  │  chisurf.fits           │  │
│  │  chisurf.imported_datasets │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
           ↑  JSON-RPC 2.0  │
           │  over ZMQ      │
           └────────────────┘
```

### ChiSurfServer (`chisurf/server/app.py`)

The server factory that wires all components:

- Creates a `ServiceDispatcher` and registers all service handlers
- Creates a `JobManager` for async job lifecycle
- Creates an `InProcessEventBus` for pub/sub events
- Creates a `ZmqServer` that binds to TCP ports for commands and events

```python
from chisurf.server.app import ChiSurfServer

server = ChiSurfServer(cmd_port=8765, pub_port=8766)
server.serve_forever()  # blocking
```

### ServiceDispatcher (`chisurf/server/dispatcher.py`)

Routes JSON-RPC method names to handler functions:

- `register(name, handler)` — register a handler
- `dispatch(method, params)` — call the handler and return a `ServiceResult`
- `has_method(name)` / `list_methods()` — introspection
- `_build_default_registry()` — registers all built-in service handlers

The dispatcher catches exceptions from handlers and returns structured
error results (`{"ok": False, "error": "..."}`) rather than crashing.

### Services (`chisurf/server/services/`)

GUI-independent service modules that operate on `chisurf` globals:

| Module | Actions | State Accessed |
|--------|---------|----------------|
| `datasets.py` | `list_datasets`, `get_dataset_info`, `add_dataset`, `remove_datasets` | `chisurf.imported_datasets`, `chisurf.fits` |
| `fits.py` | `list_fits`, `get_fit_info`, `run_fit` | `chisurf.fits` |
| `parameters.py` | `get_parameter`, `set_parameter_value`, `set_parameter_fixed`, `set_parameter_bounds` | `chisurf.fits` + model parameters |
| `projects.py` | `save_project`, `load_project`, `get_project_info` | `chisurf.project` |

All services return `ServiceResult` dicts with `{"ok": bool, ...}`.
No service module imports `chisurf.gui` or any Qt widget.

### Transport (`chisurf/server/transport/zmq.py`)

Two ZMQ socket patterns:

| Socket | Pattern | Port | Purpose |
|--------|---------|------|---------|
| REP | Request-Reply | `cmd_port` (default 8765) | Synchronous RPC calls |
| PUB | Publish-Subscribe | `pub_port` (default 8766) | Event broadcasting (fit completed, dataset added, ...) |

Messages follow JSON-RPC 2.0 format:

**Request:**
```json
{"jsonrpc": "2.0", "method": "list_datasets", "params": {}, "id": 1}
```

**Success response:**
```json
{"jsonrpc": "2.0", "result": {"ok": true, "datasets": []}, "id": 1}
```

**Error response:**
```json
{"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": 1}
```

### ChisurfClient (`chisurf/client.py`)

High-level Python client wrapping `ZmqClient`:

```python
from chisurf.client import ChisurfClient

client = ChisurfClient(host="127.0.0.1", cmd_port=8765)

# Datasets
datasets = client.list_datasets()

# Fits
fits = client.list_fits()
result = client.run_fit(fit_index=0)

# Parameters
client.set_parameter_value(fit_index=0, param_id="tau1", value=3.5)
client.set_parameter_fixed(fit_index=0, param_id="tau1", fixed=True)

# Projects
client.save_project("analysis.h5")
```

The client raises `RemoteError` on transport failures or server-side errors.

### SessionState (`chisurf/server/session.py`)

Mutable runtime state container that tracks datasets, fits, and experiments
by index and UID. Supports:

- `sync_from_globals()` — pull current state from `chisurf.fits` / `chisurf.imported_datasets`
- `to_dict()` — snapshot for serialisation
- `remove_dataset(index)` / `remove_fit(index)` — removal with UID lookup
- `find_fit_by_uid(uid)` / `find_dataset_by_uid(uid)` — stable references

### EventBus (`chisurf/server/eventbus.py`)

Thread-safe publish/subscribe with pattern matching:

```python
bus = InProcessEventBus()

# Subscribe
bus.subscribe("fit.completed", my_handler)
bus.subscribe("fit.*", wildcard_handler)    # fnmatch patterns

# Publish
bus.publish("fit.completed", {"fit_index": 0, "chi2": 1.05})
```

Events are enriched with a timestamp and topic automatically.

### JobManager (`chisurf/server/jobs.py`)

Manages async job lifecycle:

```
QUEUED ──► RUNNING ──► COMPLETED
                 │
                 └──► FAILED
                 │
                 └──► CANCELLED
```

```python
jm = JobManager()

def costly_task(params):
    import time
    time.sleep(5)
    return {"result": 42}

job_id = jm.run_fn(costly_task, {}, run_async=True)
# ... later ...
job = jm.get_job(job_id)
assert job.status == "COMPLETED"
```

Jobs support cooperative cancellation via `should_cancel(job_id)`.
Completed/failed jobs older than a configurable threshold are cleaned up.

### Dependency

The server requires `pyzmq>=25.0`. Install with:

```bash
pip install chisurf[server]
```

---

## Project Model

### Save / Load Format

Projects are saved to HDF5 files via `tables` (PyTables). The project
file bundles:

| Group | Contents |
|-------|----------|
| `/fits/` | Fit configurations, model parameters, data references |
| `/datasets/` | Dataset metadata |
| `/history/` | Action history log |
| `/extra/` | Plugin-specific metadata, action catalog snapshots |

```python
import chisurf
chisurf.project.save("analysis.h5")
chisurf.project.load("analysis.h5")
```

### Project Metadata

Action catalog metadata is embedded in `proj.extra["action_catalog"]` on save,
enabling history-aware loading and replay. The `history_loaded` flag marks
whether persisted history was restored.

---

## Data Flow Examples

### "List Datasets" (Synchronous RPC)

```
Client                          Server
  │                               │
  ├─ ZMQ REQ ─────────────────► ZMQ REP
  │  {"method":"list_datasets",   │
  │   "params":{}, "id":1}       │
  │                              ├─ Dispatcher.dispatch("list_datasets", {})
  │                              │  → datasets.list_datasets()
  │                              │  → returns {"ok":True,"datasets":[...]}
  │◄─ ZMQ REP ───────────────── ZMQ REP
  │  {"result":{"ok":True,       │
  │   "datasets":[...]}, "id":1} │
  │                              │
  client.list_datasets() → [...]
```

### "Run Fit" with Job Lifecycle

```
Client                          Server
  │                               │
  ├─ call("run_fit",{idx:0}) ──► │
  │                              ├─ fits.run_fit()
  │                              │  → fit.run() (may be long)
  │◄─ {ok:True, chi2_before:..., │
  │      chi2_after:...}         │
  │                              │
  │  (Event via PUB/SUB)         │
  │◄─ topic:"fit.completed" ──── │
  │    {fit_index:0, chi2:1.05}  │
```

### Plugin Load Flow

```
ChiSurf startup
  │
  ├─ Scan chisurf/plugins/
  ├─ Scan ~/.chisurf/plugins/
  │
  ├─ For each package:
  │    ├─ Import __init__.py
  │    ├─ Read name = "Category:Name"
  │    ├─ Load icon.png (if present)
  │    └─ Register in PluginManager
  │
  └─ User opens Plugin Manager:
       ├─ Enable plugin
       └─ __name__ == "plugin" → execute entry point
```

---

## Design Constraints

| Constraint | Rationale |
|-----------|-----------|
| **Server must not import `chisurf.gui`** | Servers may run in headless mode or subprocess without Qt |
| **Services operate on globals directly** | Initial implementation uses existing state; `SessionState` provides an abstraction layer for future migration |
| **All new code is additive** | Existing files (except `README.md`, `pyproject.toml`, `docs/`) are not modified |
| **JSON-RPC 2.0 over ZMQ** | Lightweight, no HTTP dependency, built-in pub/sub, same API for local and remote |
| **Lazy imports for heavy/unavailable deps** | Avoids crashes when optional extensions (e.g. `tttrlib`) are not installed |
