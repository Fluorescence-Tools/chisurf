# START HERE — MFDB / chisurf overhaul

_Branch: `development` (work stays on dev; **no push, no master merge** unless asked).
Authoritative plan: `MASTER-ORDER.md`. Last updated 2026-06-23._

**Phase 1 (architecture foundations) is functionally complete.** Every bug-class the
foundations targeted is fixed: vocab drift + the 39-step migration chain (PRD-19),
all `fdb_*` legacy code, the `mfdb_sample` flrCIF duplicate (+ latent `sample_type`
data-loss), real-DB test pollution (PRD-18 hermetic harness), the owner-mismatch
"Mine shows 0" class (PRD-17 canonical resolver), and a first-run production breakage
(curated DB regenerated). PRD-27's go/no-go is locked (append-only-lite).

**▶ NEXT: Phase 2 — the operation/transformer spine (PRD-11 + PRD-16).** Refactor
Burst Selection (PRD-04) and Microtime Shifter (PRD-09) as the two reference
conformant transformers; apply PRD-23 (thin widgets) as you touch them; then PRD-26
(model-driven layer). See `MASTER-ORDER.md` Phase 2.

---

## Optional Phase-1 polish (non-blocking — the bugs are already fixed)

These are the remaining DoD *checkboxes*, but each is now cleanliness, not a live
issue — do opportunistically or skip:

- **PRD-17 build-once threading.** The canonical resolver already makes reads/writes
  agree (the actual bug). Threading a `SessionContext` *object* through handler/
  `browse_datasets` signatures (vs. each calling `resolve_active_user_id`) is DI polish.
  `resolve_session(auth, db)` exists in `chisurf/core/mfdb/session.py`.
- **PRD-18 Task 4 — de-namespace `resolve_database_path`.** ~32 files do
  `from … import resolve_database_path`. This *was* the test-pollution root, but the
  hermetic harness overrides via `CHISURF_SETTINGS_DIR` *inside* `get_path`, so the
  binding is now **functionally harmless**; converting to qualified calls is pure style.
- **PRD-19 extension-declaration polish.** The `mfdb_*` provenance/object-store/vocab
  tables are declared as `mfdb_` extension categories; optionally tighten them as
  *flrCIF* extensions FK'd to flrCIF, and add live⊇declared + vocab==dictionary asserts
  to the gate.

## ▶ Phase 2 — operation/transformer spine (IN PROGRESS)

Read `MASTER-ORDER.md` Phase 2. **PRD-11** (operation-node abstraction) + **PRD-16**
(transformer contract) ship together; they **supersede PRD-07**.

**Landed:**
- ~~PRD-11 operation-parameter schema~~ ✓ (`1da6ab6d`): `.dic`-declared
  `mfdb_operation_parameter_def` (typed schema per operation type) + `mfdb_parameter.role`
  (role-indexed params).
- ~~PRD-11 seed + validate~~ ✓ (`+seed/validate`): `data/operation_parameter_defs.json`
  (authored defs for `microtime_shift` + `burst_selection`) seeded into the table on
  migrate; `operation_parameters.py` (`validate_operation_parameters` rejects
  unknown/missing-required).
- ~~PRD-16 transformer contract~~ ✓ (`+transform`): `chisurf/core/transform/`
  (`PortSpec`, `Transformer` Protocol, registry, `check_transformer_conformance`).

- ~~PRD-11 Task 3 — `register_operation`~~ ✓: records an operation node (typed
  inputs/outputs, `derived_from` edges, **role-indexed parameters** via
  `record_parameter`/`_record_parameters` `role`), validated against the `.dic`
  schema (fail-loud, no partial rows). `result_registry.register_operation`.
- ~~Microtime Shifter as a conformant transformer~~ ✓: `api/transformer.py` wraps the
  pure `shift_file`; declares ports + `operation_type`; passes conformance.
- ~~Retire `mfdb_microtime_shift`~~ ✓ (`19fbc18a`): the shifter records
  `operation_type="microtime_shift"` with the role-indexed `shift` parameter (role =
  channel); the bespoke write-only table is gone end to end (`.dic`, index,
  `add_microtime_shift`, gate; added to `_drop_legacy_tables`).
- ~~Strict validation wired into `register_result`~~ ✓ (`2a02f560`): both reference
  operation types fully declared (`burst_selection` now has all 9 params); a param
  outside an operation type's `.dic` schema raises `OperationParameterError` at the
  boundary (no partial rows); no-op for undeclared types.
- ~~Registry conformance gate~~ ✓: every registered transformer is checked
  (`test_all_registered_transformers_conform`).

- ~~Burst Selection as a conformant transformer~~ ✓: `burst_selection/api/transformer.py`
  wraps the pure `analyze_request`; `settings_from_parameters` maps the flat declared
  params back onto nested `AnalysisSettings`. **Both reference transformers now pass
  the registry conformance gate.**

**Next (in order):**
1. ~~**PRD-28 — ndXplorer ↔ MFDB burst round trip**~~ ✓ **DONE.** Both directions
   landed (direction A "Open Burst in ndXplorer" launcher + menu plugin; direction B
   "to ndXplorer" toolbar button), plus the **headless CLI handoff**: `csc
   burst-selection analyze --mfdb` registers raw+sample, burst tables, and a single
   named output-folder group whose artifact resolves (via `mfdb.datasets.open`) to the
   co-located `.bur` folder ndXplorer opens. Manual-test fixes folded in (proximity
   ratio = raw micro-time ranges, multi-file grouping, open-from-MFDB path,
   `external_reference` materialization). Spec updated in
   `PRD-28-ndxplorer-burst-integration.md`. Follow-on specs written:
   **PRD-31** (ndXplorer headless burst-filter + imaging CLI), **PRD-34** (BID saves to
   MFDB + downstream plugins ingest MFDB BIDs directly). _(Several ndXplorer-side
   usability/perf fixes also landed in the `modules/ndxplorer` submodule working
   tree — tangential to the core MFDB track.)_
2. **▶ PRD-26 (model-driven data layer) — IN PROGRESS.** The `.dic` generates the
   DAO/admin/validation/docs. **Task 1 (DAO generator core)** is the current concrete
   deliverable: `chisurf/core/mfdb/dao.py` — a dictionary-/schema-validated,
   fully-parameterised CRUD layer (`insert/get/list/update/soft_delete`) that hand SQL
   can migrate onto (Task 2). Builds on `DictionarySchemaMap` (PRD-19) + the
   operation-parameter schema (PRD-11).
3. **Apply PRD-23 (thin widgets)** to both transformer tools; consider routing
   `register_result` internals through `register_operation` so there is one recording
   path.

Orange3 lessons fold in here: typed kind-matched ports (16/11), transformer-as-
serializable-value (16/22), `compute_value`-style replayable provenance (21/27).

---

## Status snapshot

| Foundation | State | Key commits |
|---|---|---|
| **PRD-19** schema/vocab/migrations | version chain removed ✓, vocab from `.dic` ✓, **all `fdb_*` removed** ✓, legacy-free gate ✓, existing-DB column reconcile ✓, curated DB regenerated ✓, **flrCIF `mfdb_sample` collapse ✓** — minor extension-declaration polish remains | `bb837a07`, `c65f6b4a`, `638b5e69`, … |
| **PRD-18** hermetic harness | Task 1 (temp-DB redirect + guard) ✓, Task 2 (real-client integration test) ✓, existing-DB column reconcile ✓, curated DB regenerated ✓ — Task 4 (de-namespace) optional polish | `547b5a51`, `638b5e69`, … |
| **PRD-17** identity/session | canonical resolver ✓, `register_*` injection ✓, anonymous-fallback removed ✓ — build-once object-threading optional polish | `189db5a3`, `c6a73a17`, `e25cabb6` |
| **PRD-27** append-only core | go/no-go **decided: append-only-lite** ✓ (build is Phase 3) | `(PRD-27 doc)` |
| **PRD-11** operation nodes (Phase 2) | param schema ✓, `role` ✓, seed+validate ✓, `register_operation` ✓, role-indexed recording ✓, **`mfdb_microtime_shift` retired** ✓, **validation wired into `register_result`** ✓ | `1da6ab6d`…`2a02f560` |
| **PRD-16** transformer contract (Phase 2) | `PortSpec`/`Transformer`/registry/conformance ✓, **registry gate** ✓, **both reference transformers (Microtime Shifter + Burst Selection) conformant** ✓ | `(core/transform, tttr, burst)` |
| **PRD-28** ndXplorer↔MFDB burst round trip | **implemented** ✓ — both directions + headless CLI handoff (`analyze --mfdb`); manual-test fixes folded in; follow-ons PRD-31/34 spec'd | `b9137030`…`e647cc8c` |
| **PRD-26** model-driven data layer (Phase 2) | **Task 1 (DAO core) landed** ✓ — `dao.py` `DictionaryDao` (parameterised, schema-whitelisted CRUD + soft-delete), `test/fio/test_dao.py`. **Task 2 in progress** ✓ — `MFDatabase.dao` accessor + the get-by-PK family migrated (`get_artifact`/`get_operation`/`get_parameter`/`get_object_info` → `dao.get(..., include_deleted=True)`, behaviour-equivalent, A/B-verified, regression-tested). Plus the single-table `delete_*` soft-delete family (`delete_citation/probe/spectrum/setup` → `dao.soft_delete(..., deleted_at=_utc_now())`, exact marker preserved). Still deferred: `get_sample` (Row/subset shape), `delete_parameter` (dual-key). **Task 5 (docs generator) landed** ✓ — `docs_generator.py` renders tables/vocab/operation-param schemas from the `.dic`, `test/fio/test_docs_generator.py` 4/4. Task 3 already satisfied (admin fields `.dic`-driven; registry is wiring-only). **Task 4 (RPC/boundary validation from `.dic`) next** | `(core/mfdb/dao.py, docs_generator.py, repository.py)` |

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
  vocab tests; `test_dataset_browser.py` three migration/owner-backfill tests (v39 owner,
  unseeded-user, backfill-description); `test_setup_calibration_history.py::
  test_v35_migration_backfills_existing_channels`; `test_rename.py`,
  `test_becker_hickl_set.py`, `test_bhfiles.py` collection errors (hardcoded paths /
  `BeckerHicklSetReader` import). Confirm any new red is yours via a `git stash` A/B
  before owning it.
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
PRD-28 (ndXplorer round trip): `b9137030`…`e647cc8c` directions A/B + manual-test fixes
(committed earlier session); headless CLI handoff (`analyze --mfdb`) this session.
PRD-26 (model-driven layer): `0d63d1d9` `DictionaryDao` core + get-by-PK migrations
(get_artifact/operation/parameter/object_info) · `cdc5bc74` schema docs generator
(`docs_generator.py`) + DAO soft-delete value param + single-table `delete_*` migration.
Overhaul docs: `04635319` PRD-26 status / PRD-31/34 specs / PRD-28 CLI / PRD-29-33 drafts;
this commit folds the Light Path Simulator into PRD-08 and `_dev/fluorophore_db` into
PRD-06 (+ PRD-05 computed-from-optics/spectra cross-refs).
