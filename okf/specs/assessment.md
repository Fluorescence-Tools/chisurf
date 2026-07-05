---
type: Assessment
title: Cleanup Backlog — Gap vs. Target Specs
description: Concrete, verified findings where today's code diverges from the target specs, most severe first.
tags: [assessment, backlog, tech-debt, findings]
timestamp: '2026-07-05T00:00:00Z'
---

# ChiSurf — Cleanup Backlog (Gap vs. Target Specs)

> The backlog of where today's code diverges from the [target specs](index.md). Most severe first. Shrinks as the code converges.

The [specs](index.md) describe where ChiSurf should be. This document is the
gap: concrete, verified findings where the current code falls short of those
targets. It consolidates every issue surfaced while writing the specs, **plus
additional ones found by scanning the tree against the target rules**. Each
finding carries a verification status so the list can be trusted and worked down.
When this backlog is empty for a subsystem, that subsystem has reached its spec.

## Legend

**Severity** — `S1` breaks correctness or a hard invariant · `S2` contract/consistency
defect that will bite callers · `S3` tech-debt / cleanup.

**Category** — `SV` spec violation (breaks a [overview principle](overview.md#architectural-principles))
· `BUG` correctness defect · `DATA` data/schema/manifest issue · `INC` inconsistency / legacy overhang.

**Status** — `VERIFIED` re-confirmed by a scan or source read during this
assessment (evidence noted) · `REPORTED` raised by the spec-authoring pass and
recorded in the cited spec's steering notes, not independently re-run here.

## Summary

| ID | Sev | Cat | Subsystem | Finding | Status |
|----|-----|-----|-----------|---------|--------|
| [SV-01](#sv-01) | S1 | SV | Server | `chisurf.gui` imported inside the Qt-free server | ~~VERIFIED~~ ✅ FIXED |
| [SV-02](#sv-02) | S2 | SV | Server | DTO dataclasses in `dto.py` are unused; handlers hand-roll dicts | VERIFIED |
| [SV-03](#sv-03) | S2 | SV | Core/Server | Legacy `chisurf.*` globals are still de-facto shared state | VERIFIED |
| [SV-04](#sv-04) | S2 | SV | Server | `service_error` codes ride in JSON-RPC `result`, not the `error` member | REPORTED |
| [SV-05](#sv-05) | S2 | SV | Server | Event topics published ≠ topics advertised in schemas | REPORTED |
| [BUG-01](#bug-01) | S1 | BUG | Core | `NCurve(d=None)` dead branch → `np.copy(None)` | ~~VERIFIED~~ ✅ FIXED |
| [BUG-02](#bug-02) | S1 | BUG | Server | `fit_select` defined twice in `fits.py` (second shadows first) | ~~VERIFIED~~ ✅ FIXED |
| [BUG-03](#bug-03) | S1 | BUG | Server | `model_component_remove` references undefined `component_type` → `NameError` | ~~VERIFIED~~ ✅ FIXED |
| [BUG-04](#bug-04) | S2 | BUG | Core | `@abc.abstractmethod` not enforced: `Base(object)` has no `ABCMeta` | VERIFIED |
| [DATA-01](#data-01) | S1 | DATA | Plugins | **3** manifests fail validation and are silently dropped by `load_manifest()` | ~~VERIFIED~~ ✅ FIXED |
| [DATA-02](#data-02) | S2 | DATA | MFDB | `SCHEMA_VERSION = 40` is a stamp with no migration waterfall | VERIFIED |
| [DATA-03](#data-03) | S2 | DATA | MFDB | Core `mfdb_*` DDL is hand-written and defined twice (must be hand-synced) | REPORTED |
| [DATA-04](#data-04) | S2 | DATA | MFDB | `add_processing_run` partial-write; MD5 mislabeled as checksum | REPORTED |
| [INC-01](#inc-01) | S2 | INC | Core | Three overlapping instance registries with different lifetimes | REPORTED |
| [INC-02](#inc-02) | S2 | INC | Core | `@register` renames classes → fragile name-based `isinstance` | REPORTED |
| [INC-03](#inc-03) | S3 | INC | Server/MFDB | Legacy flat/`mfdb.*` aliases coexist with namespaced/`mfdb.v1.*` | REPORTED |
| [INC-04](#inc-04) | S2 | INC | MFDB | Auth enforced in ~5/40 `api.py` fns; ACL rows exist for few entity kinds | REPORTED |
| [INC-05](#inc-05) | S3 | INC | MFDB | `MFDatabase` is a ~312 KB monolith with 3 parallel access styles | REPORTED |
| [INC-06](#inc-06) | S2 | INC | Plugins | Two plugin identity conventions coexist; `ndxplorer` has no manifest | REPORTED |
| [INC-07](#inc-07) | S3 | INC | Plugins | `categories` drifts from directory group & `display_name`; demo games mixed in | REPORTED |
| [INC-08](#inc-08) | S3 | INC | Server | Generic `JobManager` bypassed by the only real long-running jobs | REPORTED |

20 findings (5 FIXED): 3 VERIFIED, 12 REPORTED. 0×S1, 11×S2, 4×S3.

---

## Spec violations (SV)

### SV-01
**S1 · Qt leaks into the Qt-free server.** Violates [overview principle 3 (headless core)](overview.md#architectural-principles)
("server MUST NOT import Qt or `chisurf.gui`") and [rpc rules](rpc.md#rules).

- Location: `chisurf/server/services/detector_setups.py:50` and `:128`.
- Evidence (scan): both lines `from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import ...`. No other `chisurf.gui` / PyQt / PySide / qtpy import exists anywhere under `chisurf/server/`.
- Impact: `python -m chisurf.server` in a headless/subprocess context will `ImportError` (or drag in Qt) the moment a `detector_setups` method runs.
- Fix: move the pure detector-setup data out of `chisurf.gui` into a Qt-free core module (e.g. `chisurf/core/...`) and import that from both the server and the widget.
- ✅ **FIXED** (2026-07-05): Created `chisurf/core/data_io/detector_setups.py` with Qt-free JSON load/save. Server imports from here. Guardrail test (`test_no_gui_import_in_server`) checks AST for `chisurf.gui` imports.

### SV-02
**S2 · DTO dataclasses are dead documentation.** Violates the [overview principle 4 (data across the boundary)](overview.md#architectural-principles)
intent and [rpc rules](rpc.md#rules).

- Location: `chisurf/server/dto.py`.
- Evidence (scan): outside `dto.py`, usage counts are `DatasetSummary` 0, `FitSummary` 0, `FitDetail` 1, `ParameterDTO` 0, `SetupDTO` 0, `ProjectInfoDTO` 0, `ActionResultDTO` 0. Service handlers build ad-hoc dicts instead.
- Impact: documented wire shapes and runtime shapes can silently drift; local-mode `ChiSurfAPI.list_fits` already returns a richer shape than server-mode `fits.list_fits`.
- Fix: either (a) make handlers construct and `.to_dict()` the DTOs so the dataclass is the single source of shape truth, or (b) delete the dataclasses and generate the contract from JSON Schemas in `server_methods.json`. Do not keep both.

### SV-03
**S2 · Legacy globals are still the de-facto shared state.** Violates [overview principle 1/2 (one door, one owner)](overview.md#architectural-principles).
The single largest source of non-uniformity across the codebase (see [core steering](core.md#steering-notes), [rpc steering](rpc.md#steering-notes)).

- Location (evidence, scan for `chisurf.fits` / `chisurf.imported_datasets` / `chisurf.experiment`): `chisurf/core/fitting/__init__.py`, `chisurf/core/models/tcspc/lifetime.py`, `chisurf/core/api/adapters.py`, `chisurf/server/startup.py`.
- Impact: the server-side `startup.py` and core model code reach around `ChiSurfAPI`/`SessionState` into process-local globals, so `local`/`hybrid`/`server` modes cannot present one consistent state.
- Fix: route these reads/writes through `ChiSurfAPI` / `SessionState`; treat the globals as a compatibility read-shim only, populated *by* the owner, never mutated directly by new code.

### SV-04
**S2 · Structured errors ride in the wrong envelope member.** [rpc rules](rpc.md#rules).

- Detail: `service_error()` results (`error_code`, `jsonrpc_code`, `exception_type`) are returned inside the JSON-RPC `result` object with `{"ok": false}`, not in the JSON-RPC `error` member. Consequently `RemoteError` only raises on transport-level failures, and every caller must still check `result["ok"]`.
- Fix: decide one contract — either promote `service_error` to the JSON-RPC `error` member (so `RemoteError` fires), or document `{"ok": bool}` as the canonical application-level result and stop implying transport errors cover it. See the spec rules for the two options.

### SV-05
**S2 · Event contract is inconsistent.** [rpc rules](rpc.md#rules).

- Detail: `parameter.*` schemas advertise a `parameter.changed` event, but no `parameter.*` handler sets `event_bus` or publishes anything; `fit.create` publishes `"fit.created"` while its schema declares `"fit.added"`; `fit.selected` / `fit.reordered` / `fit.mask_changed` / `fit.group.*` are published but undocumented.
- Fix: make `server_methods.json` the single registry of event topics and assert at startup that every published topic is declared (and vice-versa).

## Correctness bugs (BUG)

### BUG-01
**S1 · `NCurve(d=None)` mishandles the default.** [core steering](core.md#steering-notes).

- Location: `chisurf/core/curve.py:41-46`.
- Evidence (read): `if d is None: self.d = np.array([])` is immediately overwritten by the unconditional `if copy_array: self.d = np.atleast_1d(np.copy(d))`, so `d=None` runs `np.copy(None)` → an object-dtype `array(None)`, not an empty float array.
- Fix: make the `None` branch `return` or `elif`, i.e. guard the copy branch so it only runs when `d is not None`.
- ✅ **FIXED** (2026-07-05): Changed `if` to `elif` so the copy branch is guarded. Added `test_ncurve_accepts_none` regression test.

### BUG-02
**S1 · `fit_select` defined twice.** [rpc steering](rpc.md#steering-notes).

- Location: `chisurf/server/services/fits.py:483` and `:840` — the second definition silently shadows the first.
- Fix: delete the stale definition (confirm which one the dispatcher registers) and add a lint/test guard against duplicate top-level handler names in `services/`.
- ✅ **FIXED** (2026-07-05): Removed stale definition at line 483 (second at 840 has richer `_action`/`event_bus` logic). Added `test_no_duplicate_handler_names` guardrail.

### BUG-03
**S1 · `model_component_remove` raises `NameError`.** [rpc steering](rpc.md#steering-notes).

- Location: `chisurf/server/services/model_svc.py` — the fallback loop `for candidate in ("lifetimes", "species", "rotations", "distances", "gaussians", component_type):` references `component_type`, which is not a parameter of the function (signature is `state, component_index, fit_index, fit_uid, event_bus`).
- Evidence (read): confirmed the name appears only in that loop and nowhere in the function's scope.
- Impact: any call that reaches the fallback (model without `remove_component` and none of the named lists matched first) crashes with `NameError`.
- Fix: drop `component_type` from the tuple, or add it as a parameter and thread it through.
- ✅ **FIXED** (2026-07-05): Removed `component_type` from fallback tuple. Added 6 tests in `test_services_model_svc.py` covering all removal paths.

### BUG-04
**S2 · `@abc.abstractmethod` is not enforced.** [core steering](core.md#steering-notes).

- Location: `chisurf/core/base.py:242` declares `class Base(object)` — no `metaclass=ABCMeta`. Subclasses mark abstracts at `chisurf/core/parameter.py:354` and `chisurf/core/models/model.py:127,140`.
- Impact: because the MRO has no `ABCMeta`, `Parameter` and `Model` are instantiable despite their abstract methods, so a missing override fails at call time instead of construction time. (Runtime confirmation needs the `arm64` env with `chinet` built; the static class declaration is unambiguous.)
- Fix: give `Base` `metaclass=abc.ABCMeta` (or have the abstract subclasses inherit `abc.ABC`), then fix any concrete subclass that currently skips an override.

## Data / schema / manifest issues (DATA)

### DATA-01
**S1 · Three plugin manifests fail validation and are silently dropped.** Extends the plugin spec's single-manifest finding — [plugins steering](plugins.md#steering-notes).

- Evidence (ran `validate_manifest()` + `load_manifest()` over all 86 manifests):
  - `spectra_downloader/manifest.json` → `missing required field: 'id'` (uses legacy `name`/`status`/top-level `services`).
  - `kappa2_dist/manifest.json` → `rpc_methods` is `["kappa2_dist.compute"]` — a list of **strings**, not method objects.
  - `modelling/fret/manifest.json` → `rpc_methods` is six **strings**, not objects.
  - All three return `None` from `load_manifest()`, so they load only via the legacy AST fallback (or not at all).
- Root cause: `load_manifest()` catches `KeyError`/`TypeError` and returns `None`, and the discovery path never calls `validate_manifest()`, so malformed manifests are invisible at runtime.
- Fix: (1) correct the three manifests (`id` + `version`; `rpc_methods` as objects with a `name`); (2) call `validate_manifest()` during discovery and log a warning instead of silently dropping; (3) add a test that asserts every built-in manifest validates.
- ✅ **FIXED** (2026-07-05): Fixed all 3 manifests. Added validation logging in `PluginRegistry.discover()`. Added `test_builtin_manifests_all_valid` that asserts all 86+ manifests pass `validate_manifest()`.

### DATA-02
**S2 · `SCHEMA_VERSION` is a bookkeeping stamp.** [mfdb steering](mfdb.md#steering-notes).

- Location: `chisurf/core/mfdb/schema.py:14` → `SCHEMA_VERSION = 40`. There is no ordered migration waterfall behind the number; it is bumped by hand.
- Impact: a v40 stamp does not guarantee a v40 physical schema; existing user DBs cannot be reliably upgraded.
- Fix: back the version with an explicit, ordered migration list and a startup check that applies pending migrations.

### DATA-03
**S2 · Core `mfdb_*` schema is hand-written and duplicated.** [mfdb steering](mfdb.md#steering-notes).

- Detail: only `flr_*`/PDBx and six setup tables are truly dictionary-generated per the [overview principle 6](overview.md#architectural-principles) authority rule; the core `mfdb_*` tables are hand-authored DDL that exists twice — a permissive `CREATE_TABLES_SQL` and a CHECK-constrained `_CANONICAL_CHECK_SQL` — which must be kept in sync by hand.
- Fix: converge on one authoritative DDL (ideally dictionary-driven, per the invariant), and generate the CHECK-constrained form rather than maintaining a parallel copy.

### DATA-04
**S2 · `add_processing_run` partial write; MD5 mislabeled.** [mfdb steering](mfdb.md#steering-notes).

- Location: `chisurf/core/mfdb/repository.py:6385` (`add_processing_run`). The documented partial-write path can leave an operation without its artifacts/edges, and an MD5 digest is stored/labeled as a generic "checksum".
- Fix: wrap the run insertion in a single transaction (see `transactions.py`) and label the digest algorithm explicitly.

## Inconsistencies / legacy overhang (INC)

### INC-01
**S2 · Three overlapping instance registries.** [core steering](core.md#steering-notes). `Base._uuid_index`, the `@register` decorator's `_instances` set, and `chisurf/core/project/registry.py`'s `Registry` singleton track instances with different lifetimes and APIs and no single authority. → Pick one owner; make the others thin views or delete them.

### INC-02
**S2 · `@register` renames classes.** [core steering](core.md#steering-notes). `chisurf/core/decorators.py:64` sets `__class__.__name__`, which forces name-based `isinstance` fallbacks in `base.find_objects` / `find_parameters`. → Stop mutating `__name__`; match on type or an explicit tag attribute.

### INC-03
**S3 · Legacy method aliases coexist with the canonical ones.** [rpc steering](rpc.md#steering-notes), [mfdb steering](mfdb.md#steering-notes). Flat `list_datasets`/`run_fit` vs namespaced `dataset.*`/`fit.*`; MFDB `mfdb.*` (73) vs `mfdb.v1.*` (40). The prerelease `sample_database` surface is retired rather than supported for compatibility. → Remove remaining aliases directly; no compatibility schedule is required.

### INC-04
**S2 · MFDB auth is incomplete and decentralized.** [mfdb steering](mfdb.md#steering-notes), contradicts `docs/prd_mfdb_auth_rights.md`. Only ~5 of ~40 `api.py` functions check auth; ACL rows are created for essentially only `artifact` (conditionally `mfdb_operation`), so `can_access` has nothing to evaluate for samples/experiments/setups/parameters/branches; real enforcement is scattered in plugin services. → Centralize enforcement at the `api.py` boundary and create ACL rows for every guarded entity kind.

### INC-05
**S3 · `MFDatabase` is a monolith.** [mfdb steering](mfdb.md#steering-notes). ~312 KB with three parallel access styles (`repository` raw SQL, `DictionaryDao`, `orm/SampleRepository`). → Choose one access layer per concern and split the module.

### INC-06
**S2 · Two plugin identity conventions.** [plugins steering](plugins.md#steering-notes). Legacy module-level `name = "Category:Plugin"` + `if __name__ == "plugin":` vs manifest `id`/`display_name`/`entrypoints`; most plugins carry both, merged by `_read_manifest_metadata()`; `ndxplorer` is a first-class plugin with **no manifest at all**. → Make the manifest the single source of identity; backfill `ndxplorer`.

### INC-07
**S3 · Manifest metadata drifts from reality.** [plugins steering](plugins.md#steering-notes). `categories` disagrees three ways with the directory group and `display_name` (e.g. `chimol` → `["Structure","Structure","Molecular Viewer"]`; `batch_analysis` lives in `core/` but is `"Main:Tools:..."`); heavy ad-hoc `menu_hidden` for an undocumented hub/child pattern; dead `deprecated` fields; demo games (`breakout`/`pong`/`tetris`) shipped alongside production tools. → Define and enforce a category vocabulary; document the hub/child pattern or replace it; gate demo plugins behind a flag.

### INC-08
**S3 · The generic job manager is bypassed.** [rpc steering](rpc.md#steering-notes). `jobs.JobManager` exists, but the only real long-running work (fit sampling / scan) uses unlocked module-level dicts instead. → Route long-running jobs through `JobManager` so cancellation/status are uniform.

---

## How to work this list

1. ~~Land the **S1** items first — they are correctness/import failures with tiny, local fixes (BUG-01/02/03, SV-01, DATA-01), each independently shippable.~~ ✅ **Done**
2. Add the guardrails that would have caught them: manifest validation in discovery (DATA-01), a duplicate-handler-name test (BUG-02), a "no `chisurf.gui` import under `chisurf/server/`" import-lint (SV-01), and a published-vs-declared event-topic assertion (SV-05). — ✅ **DATA-01/BUG-02/SV-01 guardrails in place; SV-05 remaining**
3. Then take the **S2** structural items (SV-02/03, INC-04, DATA-02/03) as scoped refactors, each closing out the corresponding spec steering-notes entry.
4. As each finding is fixed, strike its row here and remove it from the owning spec's steering notes; when a subsystem has no findings left here, it has reached its spec.
