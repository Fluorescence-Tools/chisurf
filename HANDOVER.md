# MFDB simplification — HANDOVER

**Branch:** `development` · **HEAD:** `77f10157` · **Working tree:** clean (only the
`modules/ndxplorer` submodule shows dirty — leave it). Latest session added 9
SQL→dao commits on top of the earlier 29; all **local** (never push — see below).

This file is the start-fresh brief. The durable resume note also lives in agent
memory (`mfdb-simplification-resume.md`, indexed in `MEMORY.md`) and the full
plan at `~/.claude/plans/mfdb-is-a-bit-bubbly-sparrow.md`.

---

## 1. Environment & how to run (read first)

- **Python:** the **arm64 conda env**, NOT pixi/base:
  `PY=/Users/tpeulen/mambaforge/envs/arm64/bin/python`
- **MFDB standalone suite (the primary gate)** — proves the package is
  self-contained (no chisurf/chinet needed):
  ```
  cd modules/mfdb && PYTHONPATH=src $PY -m pytest tests -q
  ```
  **Expected: 419 passed, 1 skipped.** Run this after every change to the package.
- **ChiSurf-side integration suite:**
  ```
  PYTHONPATH="modules/mfdb/src:modules/chinet:modules/imp-tricks/src:." \
    $PY -m pytest test/fio -q -p no:cacheprovider
  ```
  Expected: **236 passed, 1 pre-existing fail** (`test_ndxplorer_cli` — needs the
  `ndxplorer` submodule built; out of scope, ignore).
- **Never `git push`.** Commit locally only. End commit messages with the
  `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.
- Working practice: material changes update the matching `okf/` concept **and**
  append a dated bullet to `okf/log.md`.

---

## 2. What the package looks like now (architecture — DONE, good)

`modules/mfdb/src/mfdb/` — top level is **5 files** (`__init__`, `api`, `config`,
`models`, `repository`) over concern subpackages:

```
schema/     schema, schema_from_dictionary, dictionary_schema_map, pdbx_metadata,
            dao (DictionaryDao), vocabulary_loader, docs_generator, _sqlutil
store/      object_store, payload_codec, payload_models, database_resolver, transactions
provenance/ lineage, graph, result_registry, compute_spec, operation_parameters
samples/    sample_manager, sample_requests, external_refs, reagents, importer, seed_data
lifecycle/  lifecycle, event_log, events, staleness
security/   auth, credentials, session, boundary_validation, base
project/    project_archiver
adapters/   chinet            (home for eLabFTW/ELN adapters — PRD-48; keep target imports lazy)
queries/    13 MFDatabase mixins (god-class breakup)  <-- SQL lives here by design
admin/      backend + cli only (RPC service; chisurf-free & import-clean)
data/       bundled .dic dictionaries + config JSON
```

- **God-class broken up:** `repository.py` went **6,397 → 1,849 lines**. `MFDatabase`
  is a thin composition of 13 mixins: `ArtifactOpsMixin` (artifact/operation/edge/
  provenance-graph core), `AnalysisMixin`, `SampleMixin`, `ProbeMixin`,
  `SetupCalibMixin`, `ObjectStoreMixin`, `ParameterMixin`, `ProtocolMixin`,
  `StudyMixin`, `BranchMixin`, `LifecycleMixin`, `ExperimentMixin`, `UserDeviceMixin`.
  What remains in `repository.py` is the legitimate core (init/connection/
  properties, migration, audit, vocabulary, experiment key-values, pdbx metadata).
- **admin GUI moved OUT** of the package to `chisurf/plugins/core/mfdb_admin/gui/`
  (it's chisurf-coupled). The admin **backend** stays in `mfdb.admin` (chisurf-free).
- **Hermetic standalone test suite** at `modules/mfdb/tests/` (isolated via
  `MFDB_SETTINGS_DIR`, no chisurf import).
- A reusable **method-extractor** for further god-class splitting is at
  `<scratchpad>/extract_mixin.py` (handles multi-line sigs/decorators, leaves
  class attrs in place, reports needed imports). Note the scratchpad path is
  session-specific; re-create from the memory note if needed.

---

## 3. THE ACTIVE TASK — eliminate scattered raw SQL (route CRUD through `db.dao`)

**Directive:** "No raw scattered SQL allowed. Use the chainsaw — mfdb is not in
production; make the arch good first, fix later." → aggressive migration; breakage
is acceptable if the suite stays green; correctness of the store matters.

**Rule of thumb:**
- CRUD (single-table insert/update/delete/select-by-key) → `db.dao.upsert / insert /
  update / soft_delete / get / list`.
- SQL in `queries/*.py` and `schema/*.py` is the **centralized home** — not
  "scattered." Priority is purging SQL from **non-query modules** (they should call
  `MFDatabase` methods, not issue SQL).
- Genuinely bespoke stays raw (flag with a `# raw ...` comment): JOINs, recursive
  graph traversal, aggregates/`DISTINCT`, refcount arithmetic, `LIKE`, and tables
  the DAO can't target (below).

### DAO limitations discovered (the suite enforces these — DON'T fight them)
- **Keyless tables** (`analysis_metadata`, `flr_sample_key_value`) → `dao.upsert`
  raises `DaoError: No primary key`. Leave raw, OR add a PK to the schema first.
- **Composite-PK junctions** (`mfdb_operation_artifact` = 4-col PK) → same. Leave raw.
- **UNIQUE-constraint upserts** need the EXACT `conflict=[...]` target or you get
  `sqlite3.IntegrityError`. Examples already fixed: `optical_properties`
  (`conflict=["probe_id","property_name"]`), `spectra`
  (`conflict=["probe_id","spectrum_type"]`). **Check `PRAGMA table_info` / the
  UNIQUE constraint before converting any `INSERT OR REPLACE`.**
- `dao.upsert(table, values, *, conflict=None, touch=True)` — omit created_at/
  updated_at (touched automatically); pass `deleted_at: None` to un-delete on
  re-add. `dao.get(table, pk_value, *, pk_column=None, include_deleted=False)`.
  `dao.list(table, *, filters=<equality dict>, order_by=, descending=, limit=, offset=)`.

### Burn-down (audit table: `okf/specs/mfdb-sql-audit.md` — keep it updated)
Package totals now: **254 select · 77 insert · 48 update · 12 delete ·
68 bespoke · 30 ddl** (down from 264/109/76/17/44/30). The **non-query-module
scattered SQL — writes AND reads — is now essentially eliminated** (api.py,
adapters/chinet.py, sample_manager, seed_data, all admin services, security/auth
are SQL-free or intentional-raw-only); the `queries/` `INSERT OR REPLACE`/
`INSERT OR IGNORE` are all converted too. What remains is legitimately the
centralized home / bespoke: `queries/` SELECTs, keyless/composite writes
(`analysis_metadata`, `mfdb_operation_artifact`, `flr_sample_key_value`),
hand-written `INSERT … ON CONFLICT DO UPDATE` (dao-equivalent), bespoke
import/merge routines, `provenance/lineage.py`+`graph.py` traversal,
`project/project_archiver.py` JOIN/count reads, and `schema/*` DDL.

**Already converted (committed, verified):** `compute_spec`, `add_entity`,
`ObjectStoreMixin` CRUD, `SampleMixin` (add_sample/update_sample/delete_sample/
condition/assembly), `UserDeviceMixin`, `ExperimentMixin.add_experiment`, probe
optical/spectra reference-import writes; **this session:** all writes in
`samples/sample_manager.py`, `samples/seed_data.py`, `admin/backend/services.py`,
`admin/backend/{password,auth,fluorophore}_services.py`, and the whole
`security/auth.py` boundary (via a schema-cached `_dao(conn)` helper). The
`bootstrap_*` seeders (`provenance/operation_parameters.py`, `lifecycle/lifecycle.py`)
are **intentional-raw** (bulk-reseed on a bare conn: aggregate reindex + delete-by-
non-PK). **DAO enhancement:** `DictionaryDao.insert` is now keyless-tolerant
(returns lastrowid on UNIQUE-only junctions like `mfdb_group_member` /
`mfdb_object_acl`) — this unblocks junction-table inserts; covered by
`test_insert_into_keyless_junction_returns_rowid`.

**Intentional-raw patterns now flagged `# raw` (leave them):** composite-key
updates/soft-deletes on PK-less UNIQUE junctions (`mfdb_group_member`,
`mfdb_object_acl`), resurrecting updates (reset `deleted_at = NULL` — `dao.update`
refuses soft-deleted rows), hard deletes (soft-delete would block re-add on a
UNIQUE), conditional/multi-column `WHERE` updates, `INSERT … SELECT` bulk copies,
`COUNT`/JOIN reads, and column-projection reads that must exclude a column
(e.g. `list_sessions` hiding `token_hash`).

### Recommended next order (the scattered-SQL core is DONE; these are cleanup)
1. `admin/seed_example.py` (13 select · 8 insert) — if it carries a `db`/dao handle,
   route its writes; it's a seeding script so several may be bulk/bespoke.
2. **Keyless/composite writes** currently raw (`analysis_metadata`,
   `mfdb_operation_artifact` 4-col PK, `flr_sample_key_value`): either leave as the
   documented DAO limitation, or add a PK/support to the DAO if you want them gone.
3. **Low-value single-table SELECTs in the mixins** → `dao.get`/`list`, only where
   trivial. Leave JOIN/aggregate/graph/DDL raw — those are the legitimate home.

### Workflow that works (learned the hard way)
- **Do NOT bulk string-replace** across varied blocks — it silently corrupts (broke
  `experiments.py` this session). Use precise per-block `Edit`s.
- Convert a file → run the **full** standalone suite → fix the exact conflict/keyless
  error it reports → commit per file (or small batch). The suite is the net.

---

## 4. Remaining lower-priority cleanup (optional; documented in memory)
- Drop `security/base.py` `MFDBClientBase` ABC only if you also retarget ~40 type-hint
  sites to `MFDatabase` (used across `result_registry` + chisurf plugins). Decided
  **skip** this session — low value, single-backend, churny/risky.
- Delete the dead hardcoded canonical DDL text in `schema/schema.py`'s
  `CREATE_TABLES_SQL` (it's overwritten at import by `_build_permissive_ddl`, so it's
  misleading-but-harmless; don't delete the whole list — it feeds `FRESH_DB_TABLES_SQL`).
- Scaffold the **eLabFTW adapter** under `adapters/` when needed (PRD-48).

---

## 5. Pointers
- Architecture concept: `okf/architecture/mfdb.md` (has the "no scattered SQL is an
  antipattern" section + package layout + queries/ mixin list).
- SQL audit / burn-down: `okf/specs/mfdb-sql-audit.md`.
- Change log: `okf/log.md` (top bullets = this session's work).
- PRDs: PRD-19 (dictionary schema), PRD-26 (model-driven/dao data layer),
  PRD-24 (standalone extraction), PRD-48 (ELN adapters).
- Author/contact: Thomas-Otavio Peulen.
