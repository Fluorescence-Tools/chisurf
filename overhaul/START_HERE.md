# START HERE — MFDB / chisurf overhaul

_Branch: `development` (work stays on dev; **no push, no master merge** unless asked).
Authoritative plan: `MASTER-ORDER.md`. Last updated 2026-06-24._

**Phase 1 (architecture foundations) is functionally complete.** Every bug-class the
foundations targeted is fixed: vocab drift + the 39-step migration chain (PRD-19),
all `fdb_*` legacy code, the `mfdb_sample` flrCIF duplicate (+ latent `sample_type`
data-loss), real-DB test pollution (PRD-18 hermetic harness), the owner-mismatch
"Mine shows 0" class (PRD-17 canonical resolver), and a first-run production breakage
(curated DB regenerated). PRD-27's go/no-go is locked (append-only-lite).

**Phase 2 (operation/transformer spine + model-driven layer) is functionally
complete:** PRD-11/16 spine ✓, both reference transformers conformant ✓, PRD-28 round
trip ✓, PRD-26 model-driven layer substantially complete ✓ (see snapshot).

**Phase 3 (provenance + LIMS) is COMPLETE: PRD-21 (lineage + events), PRD-12 (lifecycle),
PRD-14 (protocols), and PRD-13 (study/project) all landed** (see snapshot). PRD-13 added
`.dic`-declared `mfdb_study`/`mfdb_study_member`/`mfdb_study_key_value`, study CRUD +
many-to-many membership + configurable fields + `project_id` backfill, a `browse_datasets`
study facet, and a standalone admin `StudiesView`.

**Phase 4 PRD-22 (pipeline/workflow engine) COMPLETE (headless core).** `chisurf/core/pipeline/`
composes the conformant transformers into a type-checked DAG (`model.validate_pipeline`,
intersecting producer/consumer port kinds; cycles/unknown ops rejected) executed by
`runner.run_pipeline` via the PRD-21 replay-executor seam — each step a recorded operation,
queryable through the lineage API. `store.py` persists definitions + grouped runs in the
dictionary-declared `mfdb_pipeline*` tables (saveable/shareable document). burst_selection's
input now accepts `processed_data` so `raw → microtime_shift → burst_selection` type-checks.
All 3 DoD checkboxes met; Task 5 (node GUI editor) deferred to PRD-29. Tests:
`test/fio/test_pipeline.py` 11. Commits: pipeline core + persistence on `development`.

**▶ START NEXT: Phase 4 independent features** — PRD-15 (reagents), PRD-05 remainder
(gamma/crosstalk/R0), PRD-06 (fluorophore DB), PRD-08 (optics): all independent, slot as
needed. Phase-5 is PRD-24 (extract `modules/mfdb`). A natural follow-on to PRD-22 is the
node-based GUI editor (PRD-29 / Task 5) once a Qt env is in play.
Deferred LIMS threads: wire the standalone `LifecycleView`/`ProtocolsView`/`StudiesView`
into the admin dock layout once the `OVERHAUL_PLAN.md` dock rewrite lands. Standing
threads: PRD-23 "one recording path", PRD-26 upsert family. Run tests in `arm64`
(`-o addopts=""`).

**⚠️ Concurrency note (2026-06-25):** this branch is edited by multiple workers. A
concurrent commit reverted `repository.py`/`result_registry.py` to a pre-PRD-12 state
(silently breaking HEAD's committed lifecycle/protocol tests); commit `b6d22d74` restored
them. When editing those hot files, pull/commit promptly to avoid re-clobbering.

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
2. ~~**PRD-26 (model-driven data layer)**~~ ✓ **SUBSTANTIALLY COMPLETE.** The `.dic`
   now generates the DAO (`dao.py`), boundary/RPC validation (`boundary_validation.py`),
   and schema/API docs (`docs_generator.py`); admin fields were already `.dic`-driven.
   Tasks 1, 3, 4, 5 landed; Task 2 migrated the read-path, the `delete_*`/soft-delete
   family, `get_sample`, and `delete_parameter` onto the DAO. **Remaining:** the
   `INSERT OR REPLACE` `add_*`/`save_*` insert family stays hand SQL (bespoke upsert +
   explicit audit columns + per-method id-recovery — the PRD permits hand SQL for
   genuinely bespoke queries). Pick it up only if/when a DAO `upsert` primitive
   (auto audit-column management + proper `ON CONFLICT` semantics) is worth building.
3. **▶ Apply PRD-23 (thin widgets)** to both transformer tools; consider routing
   `register_result` internals through `register_operation` so there is one recording
   path. _(Next concrete deliverable; GUI-heavy — needs a Qt-capable env to smoke-test.)_

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
| **PRD-21** lineage API + event model (Phase 3) | **Complete — all DoD met.** **Task 1 (lineage read API)** ✓ `core/mfdb/lineage.py` `Lineage` (ancestors/descendants/lineage_to_root/parents/children/`what_used`/`provenance_graph`) over `mfdb_operation_artifact`; exposed on `MFDatabase` (`lineage` + `get_artifact_ancestors/_descendants/_impact/_provenance_graph`). **Task 2 (replayable compute spec)** ✓ `compute_spec.py` (`ComputeSpec`/`with_overrides`/`recompute`/`replay` + executor registry); **both reference executors wired** — `tttr_microtime_shifter/api/replay.py` and `burst_selection/api/replay.py` re-run the real pipelines (shared `db.materialize_artifact_file` → transform → register); the burst `.dic` schema was completed first (channel mask + GMM determinism) so replay is faithful. **Task 3 (event bus)** ✓ `events.py` (post-commit, best-effort, isolated; `audit_log_subscriber`) wired into `register_result`/`register_operation`. **Task 4 (calibration impact)** ✓ `what_used`/`impact_of` follow `USAGE_RELATIONSHIPS` `mfdb_edge` links to downstream artifacts. **Task 3b (admin provenance view)** ✓ already shipped in mfdb-admin (seed → upstream/downstream/full graph). Fixed an int round-trip bug (`check_value_type` accepts integral floats). Both reference transformers (microtime_shift + burst_selection) are now replayable. Tests: `test_lineage` 11, `test_events` 7, `test_compute_spec` 6, shifter+burst `test_replay` 6, `test_boundary_validation` 19. **Nothing remaining.** | `69ce1b05`…`29f0f577` |
| **PRD-13** study/project (Phase 3) | **Complete — all DoD met.** `.dic`-declared `mfdb_study` + `mfdb_study_member` (many-to-many sample/artifact membership) + `mfdb_study_key_value` (configurable fields, key-value pattern — no forked EAV). `repository`: `create_study`/`get_study`/`list_studies(scope)`, `add_study_member`/`list_study_members`/`list_studies_for_member`, `set_study_field`/`get_study_fields`, `backfill_studies_from_project_ids` (callable/idempotent). `browse_datasets` gains a `study_id` facet (direct members + sample-of-member). mfdb-admin `mfdb.studies.*` handlers + `MFDBClient` + standalone `gui/studies_view.py`. Tests: CRUD/scoping/membership/fields/browse/backfill 8, handlers 5, view 3, dataset-browser 22 = green. | `study Increments 1–3` |
| **PRD-14** protocols (Phase 3) | **Complete — all DoD met.** `.dic`-declared `mfdb_protocol` (append-only versioned; name+version, category {measurement/processing/analysis}, operation_type, setup_id, owner, is_public) + `protocol_id`/`protocol_version` on `mfdb_operation` (created/ALTERed by `reconcile_schema`). `repository`: `create_protocol` (new version on edit, never mutates), `get_protocol`/`get_protocol_by_id`/`list_protocol_versions`/`list_protocols(scope)`, `get_protocol_parameter_schema` (reuses the operation_type's PRD-11 schema — no forked stack). `register_operation(protocol_id, protocol_version)` records + validates (exists + operation_type match). mfdb-admin `mfdb.protocols.*` handlers + `MFDBClient` + standalone `gui/protocols_view.py`. Tests: CRUD/versioning 11, handlers 6, view 3 = 20. Deferred: slot `ProtocolsView` into the dock layout. | `protocol Increments 1–3` |
| **PRD-12** lifecycle state machine (Phase 3) | **Complete — all DoD met.** `.dic`-declared `mfdb_state_transition` (log) + `mfdb_state_transition_rule` (created by `reconcile_schema`); authored `data/state_lifecycle_defs.json` (sample/artifact/operation states + transitions) seeded by `lifecycle.py` `bootstrap_lifecycle_defs` into state vocabularies + rules. `repository.transition_state`/`get_state`/`get_state_history` — rule-validated (`StateTransitionError` on illegal jump), idempotent, publishes PRD-21 `state.changed` post-commit. `register_result` starts artifact/sample lifecycles best-effort. mfdb-admin `mfdb.lifecycle.*` handlers + `MFDBClient` + standalone `gui/lifecycle_view.py::LifecycleView`. Tests: schema 6, API 12, handlers 5, view 4 = 27. Deferred: slot `LifecycleView` into the dock layout (after the dock rewrite). | `lifecycle Increments 1–4` |
| **PRD-26** model-driven data layer (Phase 2) | **Substantially complete — Tasks 1, 3, 4, 5 landed; Task 2 = read/delete/get_sample/delete_parameter migrated, `insert` family is documented bespoke remainder.** **T1 (DAO core)** ✓ `dao.py` `DictionaryDao` (parameterised, schema-whitelisted CRUD + soft-delete). **T2** ✓ `MFDatabase.dao` accessor; get-by-PK family (`get_artifact`/`get_operation`/`get_parameter`/`get_object_info`→`dao.get(...,include_deleted=True)`); single-table `delete_*` soft-delete (`delete_citation/probe/spectrum/setup`→`dao.soft_delete(...,deleted_at=_utc_now())`); **`get_sample`→`dao.get(...,include_deleted=True)`** (now a dict superset; fixed two GUI callers already assuming dict `.get`); **`delete_parameter`→two `dao.soft_delete` (dual-key)**. Deferred: `INSERT OR REPLACE` `add_*`/`save_*` family (bespoke upsert + audit cols + id-recovery — PRD allows hand SQL for bespoke). **T5 (docs generator)** ✓ `docs_generator.py`. **T3** ✓ already satisfied (admin fields `.dic`-driven; registry wiring-only). **T4 (RPC/boundary validation from `.dic`)** ✓ `boundary_validation.py` `DictionaryValidator` (+shared `check_value_type`; op-param validation now type/bound-checked). Tests: `test_dao` 11/11, `test_docs_generator` 4/4, `test_boundary_validation` 17/17. | `0d63d1d9`,`cdc5bc74`,`93cca203`,`254dc26a` |

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
