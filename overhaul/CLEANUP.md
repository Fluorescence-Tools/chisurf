# ChiSurf Cleanup & Architecture Roadmap

**For an agent with zero context about this codebase.**
Every task is self-contained: read the file, make the edit, run verification.

---

Principle: clean code is **readable**, **typed**, and **maintainable**.
No dead code. No silent failures. No duplication without reason.

Test suite baseline:
```bash
python -m pytest test/server/ --tb=line -q --no-cov
```

---

## Two Tracks

This plan splits into two parallel tracks:

1. **Correctness Cleanup** — stop tests from lying, remove real duplication, fix concrete bugs.
2. **Architecture Migration** — resolve the DTO contract, reconcile docs with code, reduce global-state coupling.

**Rule**: correctness tasks come first. Architecture tasks must preserve passing tests at every step.

---

## Phase 0 — Reconcile Architecture Baseline

### Goal

Make docs and code agree before deleting anything. The migration docs (`docs/client_server_migration_plan.md`) describe adding `chisurf/api`, `PluginContext`, namespaced RPC aliases — but those already exist. The architecture doc (`docs/architecture_client_server.md`) says proxies must not be installed by default — but `chisurf/gui/__init__.py` calls `install_proxies()`.

### Tasks

0.1 Update `docs/client_server_migration_plan.md` to mark completed phases (Phase 2/3 are done).

0.2 Audit `chisurf/gui/__init__.py` proxy installation:
   - `chisurf.api.install_proxies()` replaces global lists with proxy objects
   - Architecture doc says proxies must NOT be default GUI behavior
   - **Decision**: either update the code to not install proxies by default, or update the doc to reflect that they are intentionally enabled during migration

0.3 DTO decision (see Phase 2 below) — pick one canonical approach and update both docs and code.

0.4 Verify server imports Qt-free:
```bash
python -c "import chisurf.server; print('server import OK')"
```

---

## Phase 1 — Test Truthfulness First

### Goal

No test should pass without testing its assertion. Every `if ft.get("ok"):` pattern that silently skips its body must become an explicit `pytest.skip()` or a hard assertion.

### 1.1 Fix silent-skip in `test/server/test_integration.py`

**Before** (10 occurrences):
```python
ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
if ft.get("ok"):
    # ... actual test body ...
    # If fit creation fails, this whole block is silently skipped → test PASSES
```

**After**:
```python
ft = client.fit__create(dataset_index=ds["dataset_index"], model_name="TCSPC")
if not ft.get("ok"):
    pytest.skip("TCSPC model not available in this environment")
# ... actual test body (dedented) ...
```

**Scope**: All `if ft.get("ok"):` blocks in `test/server/test_integration.py`.

**Verification**: Run tests — skipped count increases, but no false PASSES.

### 1.2 Fix silent-skip in `test/server/test_rpc_edge_cases.py`

Same pattern — find and fix `if ft.get("ok"):` occurrences.

### 1.3 Fix silent-skip in `test/server/test_integration_lifecycle.py`

Same pattern — find and fix `if ft.get("ok"):` occurrences.

### 1.4 Fix `test_build_graph_linked_parameters`

**What**: In `test/server/test_services_fits.py` (class `TestGraphService`), this test creates linked parameters but ends with `pass` and no meaningful assertion about link edges.

**Fix**: Replace the empty `pass` body with explicit assertions about node/edge counts and cross-fit link edges.

---

## Phase 2 — DTO Contract

### Goal

Resolve the contradiction: `CLEANUP.md 1.1` says delete `chisurf/server/dto.py`, but `docs/architecture_client_server.md` says explicit DTOs are the target architecture.

### Decision

**Keep `chisurf/server/dto.py`** — the dataclasses document the server contract even if no production code imports them directly. They serve as the schema reference for the JSON-safe dicts that services return.

### 2.1 Stop treating DTOs as dead code

- Keep `chisurf/server/dto.py`.
- Keep `test/server/test_dto.py` (schema coverage is useful).
- Add a docstring to `dto.py` stating: "These dataclasses document the JSON-RPC contract shapes. Service functions return plain dicts matching these schemas."

### 2.2 Update architecture docs

- In `docs/architecture_client_server.md`, clarify that DTO means "JSON-safe dict matching a defined schema", not necessarily a Python dataclass object.
- The `chisurf/server/dto.py` dataclasses are the canonical schema definition.
- Service handlers must return dicts compatible with these schemas.

---

## Phase 3 — Consolidate Test Infrastructure

### Goal

Reduce repeated flaky server setup. One shared fixture for free-port allocation, one shared fixture for server+client lifecycle where applicable.

### 3.1 Find all `_find_free_port` duplicates

Currently defined in:
- `test/server/test_integration.py` (line 11)
- `test/server/test_rpc_edge_cases.py` (line 18)
- `test/server/test_integration_lifecycle.py` (line 29)
- `test/server/test_client.py` (line 14)
- `test/server/test_app.py` (line 10)
- `test/server/test_transport_zmq.py` (line 32)

**Fix**: Move a single `_find_free_port` helper into `test/server/conftest.py` and import it everywhere.

### 3.2 Replace `port + 1` with paired-port fixture

The `zmq_server_port` fixture in conftest returns one free port. Using `zmq_server_port + 1` for the pub port is not guaranteed free.

**Fix**: Add a `zmq_server_ports` fixture that allocates two consecutive free ports using `bind(('', 0))` on each.

### 3.3 Merge `server_client` fixtures

After 3.1 and 3.2, consolidate duplicated `server_client` fixtures into `conftest.py`.

---

## Phase 4 — Remove Real Duplication

### Goal

Cleanup that improves maintainability without changing behavior.

### 4.1 Fix `PluginContext` duplicated methods

**What**: `chisurf/api/context.py` defines the same methods twice (lines 16–84 and lines 86–129). The second block has untyped signatures and shadows the first.

**Fix**: Delete the duplicate block (lines 86–129). Keep the typed version only.

### 4.2 Extract `_stats.py`

**What**: `_safe_chi2`, `_safe_chi2r`, `_safe_n_points`, `_safe_n_free`, `_collect_param_list` are duplicated in:
- `chisurf/server/services/fits.py`
- `chisurf/api/__init__.py`

**Fix**:
1. Create `chisurf/server/services/_stats.py` with canonical definitions.
2. Import from `_stats.py` in both files (and `session_svc.py`).

### 4.3 Simplify `graph.py` defensive wrappers

**What**: `_safe_str` and `_safe_float` in `chisurf/server/services/graph.py` add no value over plain `getattr`.

**Fix**: Replace `_safe_str(getattr(...))` with `str(getattr(...))`. Keep `_safe_float` only for the numeric value field.

### 4.4 Deduplicate `_find_free_port` (from Phase 3)

Already described above — ensure helpers are shared, not copied.

---

## Phase 5 — Fix Architecture Footguns

### Goal

Align actual code behavior with the client/server architecture.

### 5.1 Proxy installation policy

**Current**: `chisurf/gui/__init__.py` starts a private server subprocess during GUI startup, but does not install proxies unless explicitly configured.

**Problem**: Architecture doc says proxies must NOT be installed by default. Proxies break `isinstance`, `id()` comparisons, and attribute access patterns used in plugins and macros.

**Status**: Default proxy installation is disabled. GUI startup only calls `install_proxies()` when `server.install_proxies_on_startup: true` is configured.

**Fix**: Keep proxies opt-in. Do not install transparent proxies during normal GUI startup.

### 5.2 Fix `DatasetProxy._curve_cache` staleness

**What**: `chisurf/proxy/lists.py` `curve_data()` caches results forever with no invalidation.

**Fix**: Remove the cache (simplest) or add a `refresh` parameter.

### 5.3 `resolve_fit` → `_resolve_fit` (naming)

**What**: `chisurf/server/services/__init__.py` defines `resolve_fit()` — a private internal helper. Rename with leading underscore for clarity.

**Scope**: `__init__.py` (definition), `fits.py`, `parameters.py`, `models.py` (imports + usage).

### 5.4 Server version & protocol announcement

**What**: The server must communicate its version number and define a clear communication protocol.

**Changes**:
1. Added `PROTOCOL_VERSION = "1.0"` and `METHOD_CATALOGUE` in `chisurf/server/protocol.py`.
2. Enhanced `meta.ping` response to include `version` (from `chisurf.__version__`) and `protocol_version`.
3. Added `meta.protocol` endpoint returning `protocol_version` and the full method catalogue (grouped by namespace with descriptions).

**Verification**: `meta.ping` returns `{"ok": true, "status": "alive", "version": "...", "protocol_version": "1.0", "dataset_count": N, "fit_count": N}`. `meta.protocol` returns the full catalogue.

### 5.5 GUI server startup bounded failure handling

**What**: GUI startup could block forever after a server startup timeout because it called `proc.stderr.read(2048)` while the subprocess was still alive.

**Fix**: Added `chisurf/server/startup.py::terminate_and_collect_stderr()` and use it from GUI startup. The helper terminates the child, uses bounded `communicate()`, and returns a limited stderr sample.

**Verification**: `test/server/test_startup.py` covers live-process stderr collection and bounded output.

---

## Phase 6 — Migrate Coupling Hotspots (Incremental)

### Goal

Reduce direct mutation of `chisurf.fits` / `chisurf.imported_datasets` in macros, GUI widgets, and plugins. Route mutations through `chisurf.api` facade instead.

### Priority Order

1. `chisurf/macros/core_data.py` — appends datasets directly
2. `chisurf/macros/core_fit.py` — appends fits directly
3. `chisurf/gui/widgets/fitting/fit_list.py` — direct list manipulation
4. `chisurf/gui/widgets/experiments/widgets.py` — direct list manipulation
5. `chisurf/plugins/chisurf/globalview/wizard.py`
6. `chisurf/plugins/chisurf/batch_analysis/wizard.py`
7. `chisurf/plugins/fluorescence_decay/tr_anisotropy/wizard.py`

### Rule For Each Migrated Path

1. Confirm server endpoint exists.
2. Confirm `ChisurfClient` method exists.
3. Confirm `ChiSurfAPI` facade method exists.
4. Migrate caller to use facade.
5. Add test or import smoke.
6. Remove old local path only after all callers migrated.

---

## Phase 7 — Defer Broad Polish

### Goal

Avoid noisy churn before architecture stabilizes.

### Deferred Items

- Adding `-> None` to all test method signatures
- Pure naming-only changes (except high-value ones like `resolve_fit`)
- Large mechanical style rewrites
- Weakref event bus (no demonstrated memory leak)
- Unused import cleanup beyond obvious dead code

---

## Verification Checklist

After ALL phases are complete:

```bash
python -m pytest test/server/ --tb=line -q --no-cov 2>&1 | tail -5
python -c "import chisurf.server; print('server import OK')"
python -c "import chisurf.gui; print('gui import OK')"
```

Check no silent-skip tests remain:
```bash
grep -n "if.*get.*ok.*skip.*:" test/server/test_integration.py
grep -n "if.*get.*ok.*skip.*:" test/server/test_rpc_edge_cases.py
grep -n "if.*get.*ok.*skip.*:" test/server/test_integration_lifecycle.py
```

All should return matches showing explicit `pytest.skip()` calls.

---

## Progress Summary

| Phase | Track | Status |
|-------|-------|--------|
| 0 | Architecture | Not started |
| 1 | Correctness | ✅ Complete |
| 2 | Architecture | Decision made (keep DTOs) |
| 3 | Correctness | ✅ Complete |
| 4 | Correctness | ✅ Complete |
| 5 | Architecture | ✅ Complete |
| 6 | Architecture | Not started |
| 7 | Deferred | — |

## Legend

| Phase | Track | Focus | Risk | Test impact |
|-------|-------|-------|------|-------------|
| 0 | Architecture | Reconcile docs/code | Medium | 0 |
| 1 | Correctness | Fix lying tests | Low | Skip count increases |
| 2 | Architecture | DTO contract | Medium | 0 |
| 3 | Correctness | Test infrastructure | Low | 0 |
| 4 | Correctness | Remove duplication | Low | 0 |
| 5 | Architecture | Fix footguns | Medium | 0 |
| 6 | Architecture | Migrate hotspots | High | Incremental |
| 7 | Deferred | Broad polish | Very low | 0 |
