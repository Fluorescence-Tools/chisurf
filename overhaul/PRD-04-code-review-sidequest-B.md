# PRD-04 Sidequest B (FCS Channel Definitions) — Code Review

Review of the FCS channel-definition implementation. Source: `PRD-04-burst-pipeline.md`
"Sidequest B: FCS Channel Definitions — Same Treatment".

Reviewed (2026-06-20):

- `chisurf/gui/widgets/wizard/tttr_channeldefinition/tttr_setup_utils.py` (new shared module)
- `chisurf/core/fluorescence/fcs/channel_setups.py`
- `chisurf/plugins/fcs/fcs_channel_preset/__init__.py`
- `chisurf/core/mfdb/{schema.py, repository.py, data/mfdb_flr_ext.dic}`
- `test/models/test_detector_setups.py`,
  `chisurf/plugins/fcs/fcs_channel_preset/test/test_widgets.py`

## Verdict

**Not complete — three test failures, two of them regressions in the
already-merged detector path.** The architecture is right (shared
`tttr_setup_utils` used by *both* detector and FCS setups, structured
`mfdb_setup_fcs_pair` child table, dict-declared correlator columns, gate extended
to the FCS category), but the refactor that extracted the shared module broke
JSON saving and a monkeypatch seam, and the FCS dialog hits a non-migrated DB.

```
FAILED test/models/test_detector_setups.py::test_save_detector_setups
FAILED test/models/test_detector_setups.py::test_default_detector_setups_store_in_mfdb
FAILED chisurf/plugins/fcs/fcs_channel_preset/test/test_widgets.py::test_fcs_channel_dialog
27 passed, 3 failed
```

## What's correct (keep)

- Shared `tttr_setup_utils.py` (`SetupTypeConfig`, `setup_id_for_name(name,
  user_id, prefix)`, `save_setup_row`, `load_mfdb_setups`, `migrate_json_to_mfdb`,
  `save_setups`/`load_setups`) is used by **both** detector and FCS code — the
  no-duplication boundary is met.
- `mfdb_setup_fcs_pair` is generated from the `.dic` (placeholder + generator,
  `schema.py:881,1148`); correlator columns `n_bins`/`n_casc`/`make_fine` are
  dict-declared on `mfdb_setup`; the gate's `_SETUP_CATEGORIES` includes
  `mfdb_setup_fcs_pair` (`test_setup_prerequisites.py:161`).
- `save_setup()` persists `fcs_pairs` to the child table (`repository.py:4056`).
- `setup_id_for_name` is prefix-parameterized; external burst callers updated to
  `setup_id_for_name(selected_setup, user_id=...)` and a back-compat wrapper
  remains in `tttr_detector_setups.py`.

## 🔴 Blocker A — JSON save broken for string paths (detector regression)

`save_setups()` in the shared module does:

```python
save_path = file_path or config.canonical_file
...
save_path.parent.mkdir(parents=True, exist_ok=True)   # tttr_setup_utils.py:245
```

`file_path` is routinely a **str** (callers pass `str` paths; the JSON export /
explicit-path API is string-based). `str` has no `.parent` →
`AttributeError: 'str' object has no attribute 'parent'`. This breaks
`save_detector_setups(data, temp_path)` and the same path for FCS — i.e. the
JSON export/fallback that Sidequest B was explicitly required to preserve.

Failing: `test_save_detector_setups`.

Fix: coerce to `Path` once —
`save_path = pathlib.Path(file_path) if file_path else config.canonical_file`
(and accept `str | Path` wherever the module takes `file_path`).

## 🔴 Blocker B — FCS dialog writes to a DB missing `is_public`

`FCSChannelDialog()` construction calls `load_fcs_channel_setups()` →
`_migrate_json_to_mfdb()` → `save_setup()`, which inserts `is_public`. Against the
resolved DB this fails:

```
sqlite3.OperationalError: table mfdb_setup has no column named is_public
repository.py:3944
```

Two underlying problems:

1. **Silent migration gap.** The v32 `ALTER TABLE mfdb_setup ADD COLUMN
   is_public ...` is wrapped in `except sqlite3.OperationalError: pass`
   (`schema.py:3684-3698`). If that ALTER ever errors, the DB still advances to
   v32/v33 *without* the column — leaving exactly this "version says migrated but
   column missing" state. Swallowing the error hides a corrupt migration. Make
   the column adds idempotent-but-verified (check `PRAGMA table_info` and add if
   absent; do not silently pass on a real failure), or guarantee the columns
   before bumping the version.
2. **Non-hermetic test.** `test_fcs_channel_dialog` constructs the real dialog,
   which resolves the **real user DB** (`resolve_database_path()` →
   `{settings}/sample_management.db`) and writes to it on construction. The test
   is not isolated and mutates developer/CI state. The dialog should not perform a
   migrating MFDB write as a side effect of construction, and the test should use
   a temp DB fixture.

Failing: `test_fcs_channel_dialog`.

## 🔴 Blocker C — refactor broke the detector monkeypatch seam

`test_default_detector_setups_store_in_mfdb` patches
`detector_setups_module._db`, `._migrate_json_setups_to_mfdb`, and
`._resolve_active_user_id`, then asserts `save_detector_setups(...)` is truthy.
After the refactor, `save_detector_setups` routes through
`tttr_setup_utils.get_db` / `resolve_active_user_id`, so the module-level patches
are dead, the call falls through to the real DB path, and returns `False`.

This is refactor-induced: the indirection seam the test relied on moved. Either
(a) keep `_db` / `_resolve_active_user_id` in `tttr_detector_setups` as the
single indirection point the shared module calls, or (b) update the test to patch
`tttr_setup_utils`. Option (a) is preferable — it keeps existing call sites and
tests stable.

Failing: `test_default_detector_setups_store_in_mfdb`.

## Required before "complete"

- [x] Blocker A — FIXED: `save_setups` coerces `file_path` to `Path`
      (`tttr_setup_utils.py:225`); `test_save_detector_setups` green.
- [x] Blocker B — FIXED: schema.py v31/v32/v33 `ALTER TABLE ADD COLUMN` now
      uses `_ensure_column()` (PRAGMA table_info, add only if absent) instead of
      `try/except OperationalError: pass`; `load_fcs_channel_setups()` accepts
      `_skip_migration=True` so the dialog reads from MFDB without writing;
      `FCSChannelDialog._load_state()` passes `_skip_migration=True`; the test
      uses a temp DB via monkeypatched `resolve_database_path` and
      `resolve_active_user_id` so it never touches developer/CI state.
- [x] Blocker C — FIXED: `save_setups` accepts injectable `get_db_fn` /
      `resolve_user_fn` seams (default to the shared resolvers); `save_detector_setups`
      passes the module-level `_db` / `_resolve_active_user_id`, restoring the
      monkeypatch seam. `test_default_detector_setups_store_in_mfdb` green.
- [x] Detector + prerequisite + burst suites pass together (29 passed):
      `test_detector_setups`, `test_setup_prerequisites`, `test_setup_user_migration`,
      `test_burst_pipeline_mfdb`.
- [x] Re-run including the FCS preset tests once Blocker B is fixed.

## Fix log (A, C applied 2026-06-20; B applied 2026-06-20)

- `tttr_setup_utils.save_setups`: JSON-fallback path now
  `save_path = pathlib.Path(file_path) if file_path else config.canonical_file`
  (Blocker A); added `get_db_fn` / `resolve_user_fn` params used lazily so the DB
  and active-user seams are injectable (Blocker C).
- `tttr_detector_setups.save_detector_setups`: passes `get_db_fn=_db`,
  `resolve_user_fn=_resolve_active_user_id`.
- `schema.py`: v31/v32/v33 `ALTER TABLE ADD COLUMN` blocks replaced
  `try/except OperationalError: pass` with `_ensure_column()` which checks
  `PRAGMA table_info` and adds only if absent; same fix applied to the auth
  column adds at the bottom of `migrate_schema()`. (Blocker B, root cause 1.)
- `channel_setups.load_fcs_channel_setups()`: accepts `_skip_migration`
  parameter — when True, reads from MFDB but skips the JSON→MFDB
  migration write. (Blocker B, root cause 2, part 1.)
- `FCSChannelDialog._load_state()`: passes `_skip_migration=True` so
  construction stays read-only; migration is deferred to the first save.

## Follow-up: removed monkeypatching from the FCS test (DI instead)

The first Blocker-B fix made `test_fcs_channel_dialog` hermetic via monkeypatching
`resolve_database_path` / `resolve_active_user_id`. That was replaced with
constructor dependency injection — cleaner, and the test no longer patches
anything:

- `tttr_setup_utils.get_db(db_path=None)` — optional path override.
- `load_detector_setups(file_path=None, db_path=None, skip_migration=False)` and
  `load_fcs_channel_setups(..., db_path=None)` thread the path to `get_db`; both
  support read-only loads.
- `FCSChannelDialog(parent=None, db_path=None)` stores `db_path`; `_load_state` /
  `_reload_detector_setups` pass `db_path=self._db_path` and `skip_migration=True`
  (construction is fully read-only on both the FCS *and* detector paths).
- `test_fcs_channel_dialog` now just constructs
  `FCSChannelDialog(db_path=str(tmp_path / "sample_management.db"))` — no
  monkeypatch. The DB is an isolated temp file; read-only construction means no
  write/FK side effects, so the active-user patch is gone too.
- Backward-compat: `load_detector_setups` calls `_db(db_path) if db_path is not
  None else _db()`.

DI extended to the detector save path too (so its test drops monkeypatching as
well):

- `save_detector_setups(..., db_path=None, user_id=None)` and
  `load_detector_setups(..., db_path=None, user_id=None, skip_migration=False)`
  build `get_db_fn` / `resolve_user_fn` from the injected `db_path` / `user_id`
  (falling back to the real resolvers when not given).
- `test_default_detector_setups_store_in_mfdb` now passes
  `db_path=str(tmp_path/...)`, `user_id=""`, `skip_migration=True` — no
  `monkeypatch.setattr` of `_db` / `_resolve_active_user_id` /
  `_migrate_json_setups_to_mfdb`.

Verified: 30 passed across the FCS preset, detector, prerequisite,
user-migration, and burst suites; no monkeypatching remains in either the FCS or
detector setup tests.
  (Blocker B, root cause 2, part 1.)
- `test_widgets.test_fcs_channel_dialog`: monkeypatches
  `resolve_database_path` → `tmp_path/sample_management.db` and
  `resolve_active_user_id` → `user_default` so the test is fully hermetic.
  (Blocker B, root cause 2, part 2.)

## Note

The FCS feature work itself (schema, dict, gate, child table, ownership reuse) is
sound. The blockers are all in the **shared-module extraction** — they regress the
detector path that was previously green. Run the detector *and* FCS suites
together on every change; passing only the new FCS tests hides A/C.
