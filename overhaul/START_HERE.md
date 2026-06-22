# START HERE — MFDB / chisurf overhaul

_Branch: `development` (work stays on dev; **no push, no master merge** unless asked).
Authoritative plan: `MASTER-ORDER.md`. Last updated 2026-06-22._

We are in **Phase 1 (architecture foundations)**. Most of PRD-19, PRD-18, and PRD-17
have landed; what remains in Phase 1 is the **flrCIF table collapse** (PRD-19) and the
**identity/DI finish** (PRD-17 build-once + PRD-18 3–4), then **PRD-27**.

---

## ▶ NEXT (recommended order)

### 1. Finish the identity / DI foundation (small, in-flight — closes PRD-17 + PRD-18)

The canonical resolver and `register_*` injection are done; what's left is making the
**entry point build the `SessionContext` once** and removing fragile module-global
resolution.

- **Build `SessionContext` once at the RPC boundary.** In
  `chisurf/plugins/core/mfdb_admin/backend/services.py`, the dataset handlers already
  resolve identity via `_resolve_owner_id(db, auth)` (which delegates to the canonical
  resolver). Build one `resolve_session(auth, db)` per dispatch and pass `session=` to
  any registration call; thread it to `browse_datasets` for the read scope. Files:
  `services.py` (`datasets_browse_handler` ~263, `datasets_open_handler` ~313),
  `chisurf/core/mfdb/session.py` (`resolve_session`).
- ~~**PRD-18 Task 2 — integration test against the real in-process `MFDBClient`.**~~
  **DONE** (`test_mfdb_client_integration`, drives public `.call` browse+open). Writing
  it caught + fixed an existing-DB migrate bug (see "🔴 production follow-up" below).
- **PRD-18 Task 4 — remove namespace-bound `resolve_database_path`.** ~32 files do
  `from … import resolve_database_path`, which makes patching fragile (the original
  test-pollution root). Prefer `database_resolver.resolve_database_path()` calls or an
  injected path/`session.db`. The hermetic harness already neutralizes the *test* risk,
  so this is now cleanliness — do it incrementally, highest-traffic modules first
  (`api.py`, `result_registry.py`, `repository.py`, the plugin `backend/services.py`).
- **DoD:** identity built once at the boundary; no handler re-resolves; one integration
  test drives the real client; `grep -rn "import resolve_database_path"` trends to zero.

### 2. PRD-19 — collapse `mfdb_*` duplicates onto authoritative flrCIF (structural)

The last PRD-19 item and the biggest remaining structural cleanup before the PRD-11
spine. **flrCIF is authoritative; the `.dic` extends it — do NOT demote flrCIF to a
codec.**
- Remove the `mfdb_*` tables that *duplicate* a flrCIF concept (`_drop_legacy_tables`
  already drops `mfdb_sample`/`mfdb_experiment`; finish repointing any remaining reads
  to the authoritative `flr_*`, e.g. `bootstrap_vocabulary._field_to_item`
  `_mfdb_sample.sample_type`).
- Declare the genuine extensions (provenance graph, object store, vocab) as proper
  flrCIF extension categories in `mfdb_flr_ext.dic`, FK'd to flrCIF.
- Optionally strengthen the gate to assert live ⊇ declared + vocab == dictionary
  (the no-legacy-table half is already asserted by
  `test_fresh_db_has_no_legacy_or_duplicate_tables`).

### 3. ~~Regenerate the shipped curated DB~~ ✅ DONE (production follow-up — option B fallout)

The curated source DB (`chisurf/core/fio/mmcif/db/sample_management.db`, tracked at
lowercase `mmcif`) was pre-PRD-19 and, after the waterfall removal, hit a `put_object`
foreign-key mismatch on first run (stale `flr_sample_users` structure that
`ALTER ADD COLUMN` can't fix). **Fixed:** `build_tools/regenerate_curated_db.py`
rebuilds it on the current schema and copies the curated/demo data (probes 7,
spectra 14, optical_properties 27, samples 3, …); ran with `--replace`. Verified:
integrity ok, 0 FK violations, production-like copy+register round trip succeeds.
Re-run the script after any future schema change that the shipped DB must carry.

### 4. PRD-27 — append-only provenance/state core (decision gates PRD-12/21)

Go/no-go on append-only-lite vs full event-sourcing (recommend lite). Decide before
the Phase-3 lifecycle/events PRDs so they're built once as projections. See
`PRD-27-event-sourced-provenance-core.md`.

### Then → Phase 2
PRD-11 + PRD-16 spine (refactor burst + microtime-shifter as the two reference
transformers), then PRD-26 (model-driven layer); apply PRD-23 (thin widgets) per tool.

---

## Status snapshot

| Foundation | State | Key commits |
|---|---|---|
| **PRD-19** schema/vocab/migrations | version chain removed ✓, vocab from `.dic` ✓, **all `fdb_*` removed** ✓, legacy-free gate ✓ — **flrCIF collapse remains** (NEXT #2) | `bb837a07`, `adc6ea08`, `083d3f8c`, `c65f6b4a`, `ffca20e7` |
| **PRD-18** hermetic harness | Task 1 (temp-DB redirect + guard) ✓, Task 2 (real-client integration test) ✓, existing-DB column reconcile ✓ — **Task 4** (namespace binding) remains; 🔴 curated-DB regen (NEXT #3) | `547b5a51`, `+column-ensure` |
| **PRD-17** identity/session | canonical resolver ✓, `register_*` injection ✓ — **build-once-at-boundary remains** (NEXT #1) | `189db5a3`, `c6a73a17` |
| **PRD-27** append-only core | not started (NEXT #3) | — |

## Decisions locked

- **PRD-19 option B:** MFDB is unreleased and there is no real data yet → **pre-PRD-19
  DBs are disposable**; no forward migration (this is why the version chain was removed).
- **flrCIF is authoritative**, extended via `mfdb_flr_ext.dic` — never demoted to a codec.
- **PRD-25 N3 / option (b):** invalid links raise `LinkValidationError` *before* the
  write transaction (no partial rows); the burst pipeline reports it as a warning.
- `flr_sample.description` = display name; long text in `flr_sample.details`.

## Working-tree & test gotchas

- **Large uncommitted working tree** (~200 files: plugins/gui/build_tools WIP) — out of
  scope. When committing, **stage only your files explicitly** (`git add <files>`): the
  index has pre-staged deletions (create_icon.py, fitting_parameters.json, icon.png)
  that otherwise get swept into your commit. Verify with
  `git show --stat HEAD --format="" | grep -c "|"`.
- **Pre-existing failures (NOT yours):** `test_sample_database_plugin.py::test_json_rpc_
  versioned_services` & `::test_gui_starts_embedded_mfdb_rpc_when_unavailable`
  (AuthError / GUI bootstrap); `test_fdb_vocab_and_migration.py` two `migrated_mfdb_edge`
  vocab tests; `test_rename.py`, `test_becker_hickl_set.py`, `test_bhfiles.py`
  collection errors (hardcoded paths / `BeckerHicklSetReader` import in other
  subsystems). Confirm any new red is yours via a `git stash` A/B before owning it.
- **GUI batching hangs:** running `test_sample_database_plugin.py` batched with other
  files under `QT_QPA_PLATFORM=offscreen` can hang. Run GUI files (or individual GUI
  tests) separately.

## Quick reference

- Hermetic mfdb test run (no GUI batching):
  ```
  QT_QPA_PLATFORM=offscreen PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
    /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest -p no:cov -o addopts='' \
    test/fio/test_schema_from_dictionary.py test/fio/test_setup_prerequisites.py \
    test/fio/test_sample_manager.py test/fio/test_burst_pipeline_mfdb.py \
    test/fio/test_session_context.py
  ```
- Tests are now hermetic (PRD-18 Task 1): `CHISURF_SETTINGS_DIR` is redirected to a temp
  dir by `test/conftest.py`; a per-test guard fails if state resolves under the real
  `~/.chisurf` (`flr/sample_management.db`). Don't defeat it.
- Key files: `chisurf/core/mfdb/{session,result_registry,repository,schema,
  schema_from_dictionary,dictionary_schema_map}.py`, dictionary
  `chisurf/core/mfdb/data/mfdb_flr_ext.dic`, gate `test/fio/test_setup_prerequisites.py`.
- Plan docs: `MASTER-ORDER.md` (order), `README.md` (index),
  `MFDB-architecture-ideas.md` (A–H + I–N: I/K→19, J→26, M→27, N→25),
  `ORANGE3-lessons.md` (compute_value→21/27, typed ports→16/11, transformer-as-value→16/22).

## Completed log (condensed)

PRD-19: `447e481e` groundwork (reconcile_schema, vocab-from-`.dic`, payload split,
LinkValidationError) · `bb837a07` drop 39-step waterfall (−1862) · `ffca20e7`
legacy-free gate + drop obsolete v16→v17 test · `adc6ea08`/`083d3f8c`/`c65f6b4a`
remove all `fdb_*` (~1100 lines: repo inert dual-writes/read-fallbacks, schema
table/index defs, legacy graph traversal + RPC + manifests, obsolete migration tests).
PRD-18: `547b5a51` hermetic harness. PRD-17: `189db5a3` canonical resolver + consolidate
~6 duplicate `default_user_id` reads · `c6a73a17` thread `SessionContext` into
`register_*`. Docs: `50c22d6c` flr_sample changelog/assertion; `bb274d1b` PRD set +
MASTER-ORDER + ORANGE3-lessons.
