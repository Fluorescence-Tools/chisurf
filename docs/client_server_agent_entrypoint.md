# ChiSurf Client-Server Migration Agent Entrypoint

This is the single starting document for agents implementing ChiSurf's
client-server migration.

## Read First

1. [`client_server_next_steps.md`](client_server_next_steps.md)
2. [`architecture_client_server.md`](architecture_client_server.md)
3. [`client_server_migration_plan.md`](client_server_migration_plan.md)

Do not start implementation before reading these files.

## Mission

Move ChiSurf toward a clean client-server architecture where:

- the server owns computation and core runtime state,
- the GUI owns Qt widgets and presentation state,
- communication is only ZMQ plus JSON-RPC 2.0,
- the app remains working after every change,
- docs stay synchronized with source behavior.

## Hard Constraints

- Do not reintroduce FastMCP, HTTP, SSE, or alternate server protocols.
- Do not make the GUI depend on transparent Python object proxies.
- Do not install transparent proxies during normal GUI startup.
- Do not send Qt objects to the server.
- Keep `chisurf.server` Qt-free.
- Prefer additive APIs first, then migrate callers, then remove old local paths.

## Current Baseline

The current codebase is intentionally hybrid:

- GUI workflows still use real in-process objects in many paths.
- `chisurf.fits`, `chisurf.imported_datasets`, and `chisurf.cs` remain process-local compatibility globals.
- `chisurf.server` has its own `SessionState` and JSON-RPC services.
- `chisurf.core.api.ChiSurfAPI` exists with local, hybrid, and server modes.
- `chisurf.core.api._client.ChisurfClient` exists and is generated from client method metadata.
- DTO dataclasses and namespaced RPC methods already exist.
- Plugins and macros still need incremental migration away from direct globals.

This baseline must keep working while migration proceeds.

## First Implementation Target

Do not restart from early migration phases. Begin with the current TODOs in
`client_server_next_steps.md`:

1. Continue `meta.protocol` method schemas.
2. Continue facade migration for one workflow at a time.
3. Add GUI/server startup smoke coverage where feasible.

Before adding a new endpoint, check `chisurf/server/server_methods.json`,
`chisurf/server/client_methods.json`, and `chisurf.core.api.ChiSurfAPI` to avoid
duplicating existing functionality.

## Validation Commands

Run after each implementation step:

```bash
python -m pytest test/server/ --tb=line -q --no-cov
python -c "import chisurf.server; print('server import OK')"
python -c "import chisurf.gui; print('GUI import OK')"
```

For server subprocess smoke testing:

```bash
python -m chisurf.server --cmd-port 18765 --pub-port 18766 --host 127.0.0.1
```

Then from another process:

```python
from chisurf.core.api._client import ChisurfClient

client = ChisurfClient(cmd_port=18765, pub_port=18766)
client.connect()
print(client.meta__ping())
print(client.meta__methods())
client.close()
```

## Migration Order

Follow this order unless explicitly instructed otherwise:

1. Extend protocol schemas and tests.
2. Route one macro/facade mutation path through server mode.
3. Move one heavy computation or mutation path to server ownership.
4. Migrate one read path to DTOs.
5. Migrate one plugin to `PluginContext` / `ChiSurfAPI`.
6. Remove local fallback only after equivalent server tests and callers exist.

## Stop Conditions

Stop and report before continuing if:

- a change requires sending Qt objects over ZMQ,
- a change would install transparent proxies into normal GUI startup,
- a migration requires rewriting multiple unrelated widgets at once,
- a legacy RPC alias would be removed before all callers are migrated,
- server tests fail for reasons unrelated to the current change,
- GUI import fails after a GUI-startup-related edit.

## Definition Of Done For Each Agent Task

- The implementation follows the architecture docs.
- New server/client behavior has tests.
- `test/server/` passes or failures are documented.
- `import chisurf.server` succeeds.
- `import chisurf.gui` succeeds when relevant and environment supports Qt.
- Documentation is updated if architecture, public RPC contracts, or migration order changes.
