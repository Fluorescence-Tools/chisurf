# ChiSurf Client-Server Migration Plan

This file is a detailed implementation plan for migrating ChiSurf toward clean
client-server computation separation. It is intended as the starting point for
implementation agents.

Read this together with
[`docs/architecture_client_server.md`](architecture_client_server.md).

## Operating Rules

- Keep the application working after each change.
- Prefer additive APIs first, then migrate callers, then remove old paths.
- Keep the server Qt-free.
- Use only ZMQ/JSON-RPC for process communication.
- Do not install transparent Python object proxies into the GUI startup path.
- Add tests with each server/client change.

## Phase 1: Stabilize The Hybrid Baseline

### Goal

Make the current state explicit and safe: GUI continues to use real local
objects, server is available as a headless API, and no proxy breaks GUI code.

### Tasks

1. Confirm `gui/__init__.py` starts server but does not call `install_proxies()`.
2. Confirm `chisurf.fits` and `chisurf.imported_datasets` remain real lists in GUI process.
3. Keep `chisurf.__server_client__` available for QtConsole and future facades.
4. Ensure server subprocess terminates on GUI shutdown.
5. Verify dynamic ports prevent multiple-instance interference.

### Tests

- `python -m pytest test/server/`
- Import smoke: `python -c "import chisurf.gui"`
- Server subprocess smoke: spawn `python -m chisurf.server`, ping with `ChisurfClient`.

### Done When

- GUI import works.
- Server tests pass.
- No main GUI startup code installs proxies.

## Phase 2: Formalize DTOs And Namespaced RPC

### Goal

Make the client/server contract explicit and versionable.

### New Files

- `chisurf/server/dto.py`
- `test/server/test_dto.py`

### DTOs To Add

- `DatasetSummary`
- `DatasetDetail`
- `FitSummary`
- `FitDetail`
- `ParameterDTO`
- `SetupDTO`
- `ProjectInfoDTO`
- `ActionResultDTO`

Use simple dataclasses or typed dict helpers. They must convert to plain JSON
dicts.

### RPC Method Names

Add namespaced methods while keeping legacy aliases during migration:

| Legacy | New |
|--------|-----|
| `ping` | `meta.ping` |
| `list_methods` | `meta.methods` |
| `list_datasets` | `dataset.list` |
| `get_dataset_info` | `dataset.get` |
| `remove_datasets` | `dataset.remove` |
| `clear_datasets` | `dataset.clear` |
| `list_fits` | `fit.list` |
| `get_fit_info` | `fit.get` |
| `run_fit` | `fit.run` |
| `remove_fits` | `fit.remove` |
| `clear_fits` | `fit.clear` |
| `get_parameter` | `parameter.get` |
| `set_parameter_value` | `parameter.set_value` |
| `set_parameter_fixed` | `parameter.set_fixed` |
| `set_parameter_bounds` | `parameter.set_bounds` |
| `get_project_info` | `project.info` |
| `save_project` | `project.save` |
| `load_project` | `project.load` |

### Tests

- Every namespaced method is listed by `meta.methods`.
- Legacy aliases still pass existing tests.
- DTO serialization produces only JSON-safe values.

### Done When

- `ChisurfClient` exposes both current compatibility methods and namespaced
  methods or a namespaced facade.
- Server tests cover aliases and namespaced methods.

## Phase 3: Add `chisurf.api` Facade

### Goal

Give GUI, macros, plugins, and QtConsole a single stable API that can route to
local code now and server RPC later.

### New Package

- `chisurf/api/__init__.py`
- `chisurf/api/client.py`
- `chisurf/api/context.py`

### Design

Create a facade object:

```python
class ChiSurfAPI:
    def __init__(self, client=None, mode="hybrid"):
        self.client = client
        self.mode = mode

    def list_fits(self): ...
    def run_fit(self, fit_uid=None, fit_index=None): ...
    def list_datasets(self): ...
    def load_dataset(self, ...): ...
```

Modes:

- `local`: use current in-process objects.
- `hybrid`: local reads allowed, server commands preferred for migrated paths.
- `server`: pure client/server operation.

Set `chisurf.api` during GUI startup and expose it in QtConsole.

### Tests

- Facade can be created without GUI.
- Facade can call server methods when a client is provided.
- Facade local mode still works with simple in-process state.

### Done When

- New code can use `chisurf.api` instead of touching globals.
- QtConsole has `api` and `client` variables.

## Phase 4: Move Mutation Paths To The Facade

### Goal

Stop adding new direct mutations to `chisurf.fits` and
`chisurf.imported_datasets`. Migrate central mutation helpers first.

### Files To Migrate First

- `chisurf/macros/core_data.py`
- `chisurf/macros/core_fit.py`
- `chisurf/actions/fit_actions.py`
- `chisurf/actions/dataset_actions.py` if present
- `chisurf/actions/parameter_actions.py`

### Required Server Methods

Add methods before migrating callers:

- `dataset.load`
- `dataset.rename`
- `dataset.group`
- `dataset.ungroup`
- `fit.create`
- `fit.update`
- `fit.set_dataset`
- `fit.set_result_idx`
- `parameter.link`
- `parameter.unlink`
- `model.finalize`
- `model.set_parse_function`

### Migration Pattern

Replace direct mutation:

```python
chisurf.fits.append(fit_group)
```

with facade call:

```python
chisurf.api.add_fit(fit_config)
```

During hybrid migration, the facade may still update local lists after the
server call so existing widgets keep working.

### Tests

- Macro tests should verify both local-visible effect and server response.
- Server tests should verify state mutation in `SessionState`.

### Done When

- Central macros no longer directly mutate lists except through compatibility
  shims.
- Existing GUI workflows still work.

## Phase 5: Move Heavy Computation To Server

### Goal

Make compute-heavy operations server-owned while GUI displays results.

### Priority Order

1. Fit execution
2. Model update/finalize
3. Parameter linking and updates
4. Dataset loading
5. Project load/save

### Fit Execution

Current direct calls include:

- `self.fit.run(...)`
- `fit.run()`
- `chisurf.run("chisurf.fits[n].set_result_idx(...)")`

Target:

```python
result = chisurf.api.run_fit(fit_uid=fit_uid)
```

The server executes optimization and emits `fit.updated`. GUI refreshes the fit
view using `fit.get` or `fit.list`.

### Dataset Loading

Current direct appends include:

- `chisurf.imported_datasets.append(dataset)`
- `chisurf.imported_datasets = new_list`
- `chisurf.imported_datasets[:] = restored_datasets`

Target:

```python
dataset = chisurf.api.load_dataset(reader="tcspc", filename=path, params=params)
```

Server creates dataset and returns DTO.

### Done When

- The migrated workflows do not call heavy `.run()`, `.update()`, or file
  readers in the GUI process.
- GUI remains responsive during fit execution.

## Phase 6: Migrate Read Paths Widget-By-Widget

### Goal

Make GUI views render DTOs instead of real domain objects.

### First Widgets

Start with low-risk list/detail widgets:

1. `chisurf/gui/widgets/fitting/fit_list.py`
2. `chisurf/gui/widgets/experiments/widgets.py`
3. `chisurf/plugins/misc/f_test/f_calculator.py`
4. `chisurf/plugins/chisurf/globalview/wizard.py`

### Migration Pattern

Replace:

```python
for fit in chisurf.fits:
    name = fit.name
    chi2 = fit.chi2
```

with:

```python
for fit in chisurf.api.list_fits():
    name = fit["name"]
    chi2 = fit["chi2"]
```

Replace identity checks:

```python
if f is current_fit:
```

with UID checks:

```python
if f["uid"] == current_fit_uid:
```

### Done When

- Widget displays same information using DTOs.
- No direct `chisurf.fits` read remains in migrated widget.
- Tests or smoke paths cover widget initialization.

## Phase 7: Plugin Migration

### Goal

Plugins communicate via `chisurf.api` / `PluginContext`, not direct global state.

### Plugin Context

Add:

```python
class PluginContext:
    api: ChiSurfAPI
    client: ChisurfClient
    main_window: object
```

Pass `PluginContext` when launching plugins. Keep old plugin loading until each
plugin is migrated.

### Rules For Migrated Plugins

- No direct `chisurf.fits`.
- No direct `chisurf.imported_datasets`.
- No `chisurf.run("...")` for server-owned state.
- Use `context.api` for data/fits/parameters/project operations.
- Use `context.main_window` only for Qt UI operations.

### Priority Plugins

1. `plugins/misc/f_test/f_calculator.py` — mostly reads fit summaries.
2. `plugins/chisurf/globalview/wizard.py` — list/detail display.
3. `plugins/chisurf/batch_analysis/wizard.py` — server-action heavy.
4. `plugins/fluorescence_decay/tr_anisotropy/wizard.py` — complex fit/dataset workflow.
5. `plugins/fluorescence_decay/irf_estimator/__init__.py`.
6. TTTR plugins using `chisurf.actions.dispatch`.

### Done When

- Migrated plugin works with server DTOs.
- Plugin does not mutate global lists directly.
- Plugin tests or smoke scripts exist.

## Phase 8: Server-Owned Session Cutover

### Goal

Complete the migration: server owns state, GUI is a client.

### Preconditions

- Core macros use `chisurf.api`.
- Main fitting/dataset widgets use DTOs.
- Important plugins use `PluginContext`.
- Fit execution and dataset loading are server-side.

### Cutover Tasks

1. Stop creating authoritative fits/datasets in GUI process.
2. Make GUI startup load a session snapshot from server.
3. Replace local `chisurf.fits`/`chisurf.imported_datasets` with read-only
   compatibility views or remove usage entirely.
4. Subscribe GUI to server events and refresh views.
5. Remove local compute fallbacks once tests cover server paths.

### Done When

- GUI can run without local authoritative fit/dataset lists.
- Server state is the single source of truth.
- The GUI can be restarted and reconnect to a server session snapshot.

## Known Risk Areas

### Python Identity

Any `is`, `id()`, or `list.index(real_object)` logic must become UID-based.

### `isinstance`

Any `isinstance(dataset, DataCurve)` logic must become DTO type checks.

### `chisurf.run("...")`

Code-string execution should not mutate server-owned state. Replace with facade
methods or explicit RPC.

### Deep Mutations

Patterns like `p.link = parameter` and `fit.model.func = ...` need dedicated
server methods. Do not rely on mutating DTOs.

### Qt Objects

`chisurf.cs`, `mdiarea`, widgets, dialogs, and plot items stay in GUI process.
Never send Qt objects to the server.

## Implementation Checklist For Each Migrated Workflow

1. Identify current local-object reads and mutations.
2. Add/extend DTOs needed for the view.
3. Add server endpoint for each mutation/compute action.
4. Add `ChisurfClient` method.
5. Add `chisurf.api` facade method.
6. Migrate one caller.
7. Add/adjust tests.
8. Run server tests and GUI import smoke.
9. Document any remaining local fallback.

## Verification Commands

Use these after each migration step:

```bash
python -m pytest test/server/
python -c "import chisurf.gui; print('GUI import OK')"
python -m chisurf.server --cmd-port 18765 --pub-port 18766
```

For subprocess smoke tests, start the server, create `ChisurfClient`, call
`ping`, `meta.methods`, `dataset.list`, and `fit.list`.
