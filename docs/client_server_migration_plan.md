# ChiSurf Client-Server Migration Plan

This plan tracks the remaining migration from the current hybrid GUI/server
architecture to cleaner server-owned computation and state. It is written for
implementation agents and should be kept synchronized with `chisurf/`.

Read this with [`architecture_client_server.md`](architecture_client_server.md).

## Operating Rules

- Keep the application working after each change.
- Prefer additive server/client/facade APIs before removing local paths.
- Keep `chisurf.server` Qt-free.
- Use only ZMQ and JSON-RPC 2.0 for process communication.
- Do not install transparent Python object proxies in normal GUI startup.
- Add tests with each server/client behavior change.
- Update docs in the same change when source behavior changes.

## Status Summary

| Phase | Status | Current reality |
|-------|--------|-----------------|
| 1. Hybrid baseline | Mostly implemented | GUI/local-object compatibility remains; server can run headless; proxies are not default GUI behavior |
| 2. DTOs and namespaced RPC | Mostly implemented | DTO dataclasses exist; namespaced RPCs and legacy aliases are registered from JSON files |
| 3. API facade | Mostly implemented | `chisurf.core.api.ChiSurfAPI`, `PluginContext`, and `ChisurfClient` exist |
| 4. Mutation paths through facade | In progress | Some macros route through API in server mode; direct global mutations remain |
| 5. Heavy computation on server | In progress | RPC endpoints exist for fit/model/dataset operations; GUI workflows still use local objects in many paths |
| 6. DTO read paths widget-by-widget | In progress | Main GUI and plugins still contain direct object reads |
| 7. Plugin migration | In progress | `PluginContext` exists; many plugins still access globals directly |
| 8. Server-owned session cutover | Not complete | GUI globals remain authoritative for non-migrated workflows |

## Implemented Foundation

The following should be treated as existing infrastructure, not future work:

- `chisurf/server/dto.py` contains `DatasetSummary`, `DatasetDetail`, `FitSummary`, `FitDetail`, `ParameterDTO`, `SetupDTO`, `ProjectInfoDTO`, and `ActionResultDTO`.
- `chisurf/server/server_methods.json` registers legacy and namespaced RPC methods.
- `chisurf/server/client_methods.json` defines generated `ChisurfClient` convenience methods.
- `chisurf/server/protocol.py` exposes `PROTOCOL_VERSION`, `METHOD_CATALOGUE`, and initial `METHOD_SCHEMAS`.
- `meta.ping`, `meta.methods`, and `meta.protocol` are available.
- `chisurf.core.api.ChiSurfAPI` supports local, hybrid, and server modes.
- `chisurf.core.api.context.PluginContext` exists for migrated plugins.
- `chisurf.core.api._client.ChisurfClient` and `RemoteError` exist.
- Structured service errors are centralized through `chisurf.server.services.service_error()`.

## Current RPC Inventory

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

Legacy aliases remain for compatibility. Do not remove them until all callers
are migrated and tests prove the namespaced methods cover the same behavior.

## DTO Policy

`chisurf/server/dto.py` dataclasses document the JSON contract shapes and provide
helpers. Service handlers may return plain dictionaries. Those dictionaries must
be JSON-safe and should match the DTO shapes where a DTO exists.

When adding or changing service result shapes:

1. Add or update the DTO/dataclass or schema documentation.
2. Return JSON-safe dicts from the service.
3. Add tests for serialization and representative service output.
4. Update `METHOD_SCHEMAS` in `chisurf/server/protocol.py` where applicable.
5. Update these docs if the public contract changes.

## Next Work: Protocol Schemas

`METHOD_SCHEMAS` currently covers the initial protocol metadata and selected
dataset/fit methods. Continue adding schema metadata incrementally.

Priority order:

1. `parameter.*`
2. `project.*`
3. `session.*`
4. `model.*`
5. `graph.*`

Each schema entry should document:

- required params
- optional params
- result shape name or DTO reference
- event topics emitted

Verification:

```bash
python -m pytest test/server/test_protocol.py test/server/test_integration.py::TestMetaProtocol --tb=line -q --no-cov
```

## Next Work: Facade Migration

Goal: reduce direct mutation of `chisurf.fits` and `chisurf.imported_datasets`.

Known remaining mutation patterns include:

- `chisurf.imported_datasets.append(...)`
- `chisurf.imported_datasets = ...`
- `chisurf.imported_datasets.clear()` / `.extend(...)`
- `chisurf.fits.append(...)`
- direct fit/model/parameter mutation through local objects
- `chisurf.run("...")` for state changes

Migration pattern:

1. Identify one workflow or caller.
2. Confirm or add the server RPC endpoint.
3. Confirm or add the `ChisurfClient` method.
4. Confirm or add the `ChiSurfAPI` facade method.
5. Route the caller through `ChiSurfAPI` in server mode.
6. Preserve local/hybrid behavior until all dependent GUI paths are migrated.
7. Add tests for server mode and compatibility behavior.
8. Update docs if the public contract changed.

Do not migrate unrelated widgets or plugins in one change.

## Next Work: Read-Path Migration

GUI views should gradually render DTOs rather than real domain objects.

Replace direct reads:

```python
for fit in chisurf.fits:
    name = fit.name
    chi2 = fit.chi2
```

with facade reads:

```python
for fit in api.list_fits():
    name = fit["name"]
    chi2 = fit["chi2"]
```

Replace identity checks:

```python
if fit is current_fit:
    ...
```

with UID checks:

```python
if fit_summary["uid"] == current_fit_uid:
    ...
```

## Next Work: Plugin Migration

Migrated plugins should use `PluginContext` or `ChiSurfAPI` for server-owned
state.

Rules for migrated plugins:

- Do not directly mutate `chisurf.fits`.
- Do not directly mutate `chisurf.imported_datasets`.
- Do not use `chisurf.run("...")` for server-owned state changes.
- Use `context.api` for data, fit, parameter, project, model, session, and graph operations.
- Use `context.main_window` only for local Qt UI operations.

## Risk Areas

| Risk | Required migration pattern |
|------|----------------------------|
| Python identity (`is`, `id`, `list.index(real_object)`) | Use stable `uid` fields |
| `isinstance(dataset, DataCurve)` | Use DTO type/name fields or explicit schema fields |
| Deep mutations (`p.link = parameter`, `fit.model.func = ...`) | Add explicit RPC/facade methods |
| Local `.run()` / `.update()` calls | Route through `fit.run`, `fit.update`, or facade methods |
| Qt objects | Keep in GUI process; never send over RPC |

## Verification Commands

Run after meaningful server/client/facade changes:

```bash
python -m pytest test/server/ --tb=line -q --no-cov
python -c "import chisurf.server; print('server import OK')"
python -c "import chisurf.gui; print('gui import OK')"
```

For subprocess smoke testing:

```bash
python -m chisurf.server --cmd-port 18765 --pub-port 18766 --host 127.0.0.1
```

Then from another process:

```python
from chisurf.core.api._client import ChisurfClient

client = ChisurfClient(cmd_port=18765, pub_port=18766)
client.connect()
print(client.meta__ping())
print(client.meta__protocol())
client.close()
```

## Stop Conditions

Stop and ask before continuing if a migration would:

- require importing Qt or `chisurf.gui` from `chisurf.server`
- install proxies during normal GUI startup
- remove a legacy alias still used by known callers
- rewrite multiple unrelated widgets/plugins in one change
- remove local/hybrid fallback before equivalent server tests exist
