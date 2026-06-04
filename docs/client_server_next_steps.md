# ChiSurf Client-Server Next Steps

This file is a handoff for an agent with no prior conversation context. Start here, then read:

1. `docs/architecture_client_server.md`
2. `docs/client_server_migration_plan.md`
3. `CLEANUP.md`

## Current State

- Transport is ZMQ REQ/REP for commands plus ZMQ PUB/SUB for events.
- Wire protocol is JSON-RPC 2.0.
- `chisurf.server.protocol.PROTOCOL_VERSION` is `"1.0"`.
- `meta.ping` returns ChiSurf version and protocol version.
- `meta.protocol` returns a namespace-grouped method catalogue.
- `ChisurfClient.meta__protocol()` exists.
- GUI startup launches a private server subprocess on dynamic ports.
- GUI startup does not install transparent proxies by default.
- Transparent proxies remain available only when explicitly enabled with `server.install_proxies_on_startup: true`.
- Server startup timeout handling is bounded by `chisurf.server.startup.terminate_and_collect_stderr()`.

## Hard Constraints

- Keep `chisurf.server` Qt-free.
- Use only ZMQ + JSON-RPC for process communication.
- Do not reintroduce FastMCP, HTTP, SSE, or alternate protocols.
- Do not install transparent Python object proxies in normal GUI startup.
- Prefer additive RPC/client/facade APIs before removing old local paths.
- Keep tests passing after every change.

## Validation Commands

Run after every meaningful change:

```bash
python -m pytest test/server/ --tb=line -q --no-cov
python -c "import chisurf.server; print('server import OK')"
python -c "import chisurf.gui; print('gui import OK')"
```

For subprocess startup smoke testing:

```bash
python -m chisurf.server --cmd-port 18765 --pub-port 18766 --host 127.0.0.1
```

Then from another process:

```python
from chisurf.client import ChisurfClient
client = ChisurfClient(cmd_port=18765, pub_port=18766)
print(client.meta__ping())
print(client.meta__protocol())
client.close()
```

## Completed Recently

- Fixed silent-skip tests and graph assertions.
- Removed duplicate `PluginContext` methods.
- Extracted duplicated stats helpers to `chisurf/server/services/_stats.py`.
- Renamed internal `resolve_fit()` to `_resolve_fit()`.
- Fixed `DatasetProxy.curve_data(refresh=False)` cache staleness.
- Consolidated free-port test helpers into `test/server/helpers.py`.
- Added `PROTOCOL_VERSION`, `METHOD_CATALOGUE`, `meta.protocol`, and versioned `meta.ping`.
- Fixed hardcoded ZMQ request ID `1`; IDs now increment per client.
- Made protocol `_next_id()` thread-safe.
- Added transport invalid-request handling for malformed requests.
- Added tests for startup stderr handling, protocol metadata, request IDs, and invalid requests.
- Made GUI proxy installation opt-in during server startup.
- Added backward-compatible structured service error metadata (`error_code`, `jsonrpc_code`, optional `exception_type`).
- Added initial method-level schemas to `meta.protocol` for the `meta`, `dataset`, and `fit` namespaces.
- **Phase 6 facade migration begun**: Added `group_datasets()` and `ungroup_datasets()` to `ChiSurfAPI` facade.
- **Server-mode routing added** to `core_data.py` `remove_datasets`, `group_datasets`, and `ungroup_datasets` — when `api.mode == "server"`, these now route through the API facade instead of dispatching locally.
- **Structured service error metadata**: Added `service_error()` helper and error code constants (`NOT_FOUND`, `INVALID_INPUT`, `OPERATION_FAILED`, `INVALID_STATE`) to `chisurf/server/services/__init__.py`. All service files (datasets, fits, parameters, models, projects) now return structured errors with `error_code` and `jsonrpc_code` instead of raw `{"ok": False, "error": ...}` dicts.
- **`RemoteError` enhanced**: Now carries `error_code`, `jsonrpc_code`, `exception_type`, and `details` attributes for structured client-side error inspection.
- **`_service_error` moved**: From `dispatcher.py` into shared `services/__init__.py` so all service functions and the dispatcher use the same helper.

## Next TODOs, In Order

### 1. Reconcile Documentation With Current Code

Files:

- `docs/client_server_migration_plan.md`
- `docs/architecture_client_server.md`
- `CLEANUP.md`

Tasks:

- Mark migration Phase 1, Phase 2, and Phase 3 as mostly implemented where accurate.
- Update the API naming table to include `meta.protocol`, `graph.build`, `graph.build_fits`, `model.finalize`, and `model.set_parse_function`.
- Clarify DTO policy: `chisurf/server/dto.py` dataclasses are schema documentation; services return JSON-safe dicts matching those shapes.
- Keep the statement that GUI startup must not install proxies by default.

Verification:

```bash
python -m pytest test/server/test_protocol.py test/server/test_dto.py --tb=line -q --no-cov
```

### 2. RPC Error Compatibility — Phase Complete

Current behavior has two layers:

- Transport errors use JSON-RPC error envelopes: `{"error": {"code": ..., "message": ...}}`.
- Service errors return `{"ok": false, "error": "...", "error_code": "...", "jsonrpc_code": ...}` inside JSON-RPC `result` for backward compatibility.

#### What was done

- **`service_error()` helper** moved to `chisurf/server/services/__init__.py` with error code constants (`NOT_FOUND`, `INVALID_INPUT`, `OPERATION_FAILED`, `INVALID_STATE`).
- **All service files** (`datasets.py`, `fits.py`, `parameters.py`, `models.py`, `projects.py`) now return structured errors with `error_code` and `jsonrpc_code` instead of raw `{"ok": False, "error": ...}`.
- **`RemoteError` enhanced**: Now carries `error_code`, `jsonrpc_code`, `exception_type`, and `details` attributes.
- **12 new tests** cover structured error metadata end-to-end (service error helper, parameter not found, fit not found, invalid input, invalid state, RemoteError attributes).
- **`_service_error`** removed from `dispatcher.py` — both dispatcher and services use the shared helper.

#### Status: Complete — no further work needed on this item.

Existing `ok/error` backward compatibility is preserved. Service failures can be mapped to JSON-RPC error envelopes in a future phase if desired, but this is not currently required.

Verification:

```bash
python -m pytest test/server/test_dispatcher.py test/server/test_client.py test/server/test_rpc_edge_cases.py --tb=line -q --no-cov
```

### 3. Continue Parameter Schemas In `meta.protocol`

Current `METHOD_CATALOGUE` lists method names and namespace descriptions. `METHOD_SCHEMAS` now covers `meta.*`, `dataset.*`, and `fit.*`.

Add the remaining method-level schema metadata incrementally:

- Required params
- Optional params
- Result shape name or DTO reference
- Event topics emitted

Next namespaces: `parameter.*`, `project.*`, `session.*`, `model.*`, and `graph.*`.

Verification:

```bash
python -m pytest test/server/test_protocol.py test/server/test_integration.py::TestMetaProtocol --tb=line -q --no-cov
```

### 4. Continue Phase 6 Facade Migration

Goal: reduce direct mutation of `chisurf.fits` and `chisurf.imported_datasets`.

#### Status: In progress — first batch complete

**Completed in `chisurf/macros/core_data.py`:**
- Added `group_datasets()` and `ungroup_datasets()` to `ChiSurfAPI` facade (in `chisurf/api/__init__.py`).
- `remove_datasets`, `group_datasets`, `ungroup_datasets` now route through `chisurf.api` when `api.mode == "server"`.
- `_from_controller` dispatch pattern preserved for hybrid/local mode.
- All 495 server tests pass; server and GUI imports OK.

**Remaining direct mutations in `core_data.py`:**

- `chisurf.imported_datasets.append(...)` — in `add_dataset` (lines 530-534) and `restore_global_fit_dataset` (line 51).
- `chisurf.imported_datasets = ...` — in `group_datasets` (line 150) and `ungroup_datasets` (line 218) local path, `remove_datasets` (line 327) local path, `reinitialize_application` (line 654).
- `chisurf.imported_datasets.clear()` / `.extend(...)` — in `reinitialize_application` (lines 654-655).

#### Next: Extend server-mode routing to remaining paths

1. Add `dataset.restore_global_fit` server endpoint for `restore_global_fit_dataset` (currently only dispatched through local action system).
2. Route `restore_global_fit_dataset` through API facade in server mode.
3. Route `add_dataset` local fallback path through API in hybrid mode.

Migration pattern:

1. Confirm server endpoint exists.
2. Confirm `ChisurfClient` method exists.
3. Confirm `ChiSurfAPI` facade method exists.
4. Route the macro through `chisurf.api` where safe.
5. Preserve local-visible behavior during hybrid mode.
6. Add tests before removing any old local path.

Do not migrate multiple widgets/plugins in one change.

Verification:

```bash
python -m pytest test/server/ --tb=line -q --no-cov
python -c "import chisurf.gui; print('gui import OK')"
```

### 5. Add GUI Startup Smoke Coverage If Feasible

Current covered piece:

- `test/server/test_startup.py` covers non-blocking failed subprocess stderr collection.

Potential next tests:

- A subprocess startup smoke test that starts `python -m chisurf.server`, pings `meta.ping`, then terminates.
- A settings-level test confirming default `server.install_proxies_on_startup` is false or absent.

Avoid importing full Qt GUI in unit tests unless the environment is known to support it.

## Stop Conditions

Stop and ask before continuing if:

- A change requires Qt imports in `chisurf.server`.
- A change would install proxies during default GUI startup.
- A migration would require rewriting multiple unrelated widgets at once.
- Tests fail for reasons unrelated to the current change.
- `python -c "import chisurf.gui"` fails after a GUI-startup-related edit.
