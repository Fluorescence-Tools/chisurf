# ChiSurf Client-Server Migration Agent Entrypoint

This is the single starting document for agents implementing ChiSurf's
client-server migration.

## Read First

1. [`client_server_next_steps.md`](client_server_next_steps.md)
2. [`architecture_client_server.md`](architecture_client_server.md)
3. [`client_server_migration_plan.md`](client_server_migration_plan.md)

Do not start implementation before reading both files.

## Mission

Move ChiSurf toward a clean client-server architecture where:

- the server owns computation and core runtime state,
- the GUI owns Qt widgets and presentation state,
- communication is only **ZMQ + JSON-RPC 2.0**,
- the app remains working after every change.

## Hard Constraints

- Do not reintroduce FastMCP, HTTP, SSE, or alternate server protocols.
- Do not make the GUI depend on transparent Python object proxies.
- Do not install `chisurf.proxy.install_proxies()` during GUI startup.
- Do not send Qt objects to the server.
- Keep `chisurf.server` Qt-free.
- Prefer additive APIs first, then migrate callers, then remove old local paths.

## Current Baseline

The current codebase is intentionally hybrid:

- GUI still uses real in-process objects:
  - `chisurf.fits`
  - `chisurf.imported_datasets`
  - `chisurf.cs`
- GUI starts a private ZMQ server subprocess on dynamic ports.
- Server has its own `SessionState` and JSON-RPC services.
- Proxy classes exist for headless/experimental use only.
- Plugins and macros still mostly use local globals.

This baseline must keep working while migration proceeds.

## First Implementation Target

Start with **Phase 2** from `client_server_migration_plan.md`:

1. Add formal DTO helpers in `chisurf/server/dto.py`.
2. Add tests in `test/server/test_dto.py`.
3. Add namespaced RPC aliases while keeping legacy methods:
   - `meta.ping`
   - `meta.methods`
   - `dataset.list`
   - `dataset.get`
   - `fit.list`
   - `fit.get`
   - `fit.run`
   - `parameter.get`
   - `project.info`
4. Update `ChisurfClient` with matching namespaced or facade-style methods.
5. Keep all existing tests passing.

Do not migrate widgets/plugins until the DTO and namespaced RPC layer is tested.

## Validation Commands

Run after each implementation step:

```bash
python -m pytest test/server/
python -c "import chisurf.gui; print('GUI import OK')"
```

For server subprocess smoke testing:

```bash
python -m chisurf.server --cmd-port 18765 --pub-port 18766
```

Then from another process:

```python
from chisurf.client import ChisurfClient
client = ChisurfClient(cmd_port=18765, pub_port=18766)
print(client.ping())
print(client.list_methods())
```

## Migration Order

Follow this order unless explicitly instructed otherwise:

1. Stabilize hybrid baseline.
2. Add DTOs and namespaced RPCs.
3. Add `chisurf.api` facade.
4. Move central macro/action mutation paths to facade.
5. Move heavy computation to server.
6. Migrate read paths widget-by-widget.
7. Migrate plugins to `PluginContext` / `chisurf.api`.
8. Cut over to server-owned session.

## Stop Conditions

Stop and report before continuing if:

- a change requires sending Qt objects over ZMQ,
- a change would install transparent proxies into normal GUI startup,
- a migration requires rewriting multiple unrelated widgets at once,
- server tests fail for reasons unrelated to the current change,
- GUI import fails.

## Definition Of Done For Each Agent Task

- The implementation follows the architecture docs.
- New server/client behavior has tests.
- `test/server/` passes.
- `import chisurf.gui` succeeds.
- Documentation is updated if the architecture or migration order changes.
