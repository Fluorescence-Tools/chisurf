# MFDB — HANDOVER

**Branch:** `development` · **HEAD:** `24f49c0f` · all commits **local** (never push).

This is the start-fresh brief. Two big MFDB threads have **landed** since this file was
first written — the SQL→DAO migration (PRD-26; see §3, now historical) and a new
**pluggable authentication** layer (PRD-59) + api.py auth enforcement (INC-04). A third,
the **`mfdb.admin` → `mfdb_admin` package extraction, is IN FLIGHT in the working tree**
(uncommitted). See §0 for what's next.

---

## 0. What's next (read first)

**⚠️ In-flight, uncommitted refactor:** `mfdb.admin` is being extracted into a new
standalone package `modules/mfdb-admin/src/mfdb_admin/`. The old `mfdb/admin/*` files are
now `from mfdb_admin… import *` wrappers, and `chisurf/plugins/core/mfdb_admin/*` re-exports
from it. This is **not committed** — finish/land it before large new work. Because of it,
the test path now needs `mfdb-admin/src` (see §1). Watch: `import *` wrappers don't
re-export underscore-prefixed names (`_validate_mfdb_methods_in_manifest` broke collection
once — fixed). GUI (`mfdb-admin/src/mfdb_admin/gui/`) + `pyproject.toml`/`README` are still
untracked.

**Prioritized next steps:**
1. **Land the `mfdb-admin` extraction** — commit it coherently; add its `pyproject.toml`
   to the workspace; confirm both suites green with the new path.
2. **INC-04 follow-ups** (auth enforcement, PRD-59-adjacent): extend owner ACLs to
   `setups`/`parameters`/`branches` on write (samples/experiments/artifacts/operations
   already do); add **v1-dispatcher round-trip tests** (drive `mfdb.v1.*` through the
   dispatcher with a token, not just the api functions directly).
3. **Lower-priority cleanup** (see §4): drop `security/base.py` ABC; delete dead
   `CREATE_TABLES_SQL` text; `boundary_validation.py` lint debt (D102 ×); local-auth
   user-enumeration timing (deliberately skipped — low value).

**No further auth providers are planned** beyond local + LDAP (no OIDC/SAML/external-IdP).

**Auth architecture (for context):** `security/auth_providers.py` = `AuthProvider` protocol +
`LocalAuthProvider`/`LdapAuthProvider`; `security/login.py` = `resolve_provider` (fixed local/ldap
dispatch) + `login()` orchestrator (authenticate → resolve/JIT-provision via
`flr_sample_users.auth_provider`/`external_id` → group reconcile → session). `api.py` threads
`auth` through all `mfdb.v1.*` functions with graceful ACL enforcement. Concept: **PRD-59**
(`okf/prds/prd-59.md`). Headless CLI: `mfdb-admin auth login|whoami|status`.

---

## 1. Environment & how to run (read first)

- **Python:** the **arm64 conda env**, NOT pixi/base:
  `PY=/Users/tpeulen/mambaforge/envs/arm64/bin/python`. `ldap3` is installed there for the
  LDAP tests (optional `[ldap]` extra for real installs).
- **MFDB standalone suite (the primary gate)** — note the **`mfdb-admin/src` path** now
  required (admin extraction):
  ```
  cd modules/mfdb && PYTHONPATH=src:../mfdb-admin/src $PY -m pytest tests -q
  ```
  **Expected: 466 passed, 1 skipped.** Run after every change to the package.
- **ChiSurf-side integration suite** (also add `mfdb-admin/src`):
  ```
  PYTHONPATH="modules/mfdb/src:modules/mfdb-admin/src:modules/chinet:modules/imp-tricks/src:." \
    $PY -m pytest test/fio -q -p no:cacheprovider
  ```
  Expected: **236 passed, 1 pre-existing fail** (`test_ndxplorer_cli` — needs the
  `ndxplorer` submodule built; out of scope, ignore).
- **Never `git push`.** Commit locally only. End commit messages with the
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
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
- **PK-less tables with a UNIQUE** (`analysis_metadata`/`UNIQUE(analysis_id,key)`,
  `flr_sample_key_value`, `flr_experiment_key_value`, `analysis_data`) → **now
  convertible**: `dao.upsert` was made keyless-tolerant (its final `primary_key()`
  return-value lookup is wrapped in `try/except → lastrowid`, mirroring
  `dao.insert`). Pass the UNIQUE as the explicit `conflict=[...]`. Only a table with
  **neither** a PK **nor** a targetable UNIQUE is a genuine `upsert` limitation.
- **Truly constraint-less tables** (`citeulike`, `products`, `standards`,
  `product_categories`) have no PK *and* no UNIQUE in the reconciled schema →
  `INSERT OR REPLACE` was silently a plain insert (dup ids possible). Converted to
  `dao.insert`; a real upsert needs a schema PK first (see §4 follow-up).
- **Composite-PK junctions** (`mfdb_operation_artifact` = 4-col PK) → **convertible
  after all**: `dao.upsert(conflict=[<all PK cols>])`. When the values are exactly
  the PK columns it's an idempotent `DO NOTHING` (identity-preserving, unlike
  `INSERT OR REPLACE`). `primary_key()` returns the first PK col and does not raise,
  so composite PKs never hit the "No primary key" path. Only *keyless* tables with
  no targetable UNIQUE are a genuine `upsert` limitation.
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
Package totals now: **151 select · 52 insert · 48 update · 12 delete ·
95 bespoke · 30 ddl** (down from 264/109/76/17/44/30). The **non-query-module
scattered CRUD is fully eliminated** (api.py, adapters/chinet.py,
sample_manager, seed_data, seed_example, all admin services, security/auth
are SQL-free or intentional-raw-only), **no `INSERT OR REPLACE` remains anywhere
in the package**, and the **trivial single-table SELECTs in `queries/*.py`
*and* `repository.py` have been swept onto `dao.get`/`list`** (select 241 → 151;
the bespoke rise 44 → 95 over the migration is mostly reclassification of
JOIN/aggregate/subquery/source/merge reads that the hand-audit had miscounted as
plain selects, not new raw SQL). A pile of **schema-mismatched dead code** was
deleted along the way (citeulike/products/standards/product_categories +
chem_descriptors/images methods). What remains is legitimately the centralized
home / bespoke:
`queries/` JOIN/aggregate/`DISTINCT`/`json_extract`/compound-`ORDER BY` SELECTs,
`bootstrap_*` bulk seeders on a bare conn (incl. `schema.py` group-member
`INSERT OR IGNORE`), hand-written `INSERT … ON CONFLICT DO UPDATE`
(dao-equivalent), the append-only `mfdb_parameter`/`mfdb_audit_log` core inserts,
bespoke import/merge routines, `provenance/lineage.py`+
`graph.py` traversal, `project/project_archiver.py` JOIN/count reads, and
`schema/*` DDL. (Composite-PK junctions are no longer a remaining item —
they convert via `dao.upsert(conflict=[<all PK cols>])`.)

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
1. ~~`admin/seed_example.py`~~ — **DONE**, now **zero raw SQL** (including the two
   composite-PK `mfdb_operation_artifact` writes, via `dao.upsert(conflict=[4 PK cols])`).
2. **Keyless writes** currently raw (`analysis_metadata`, `flr_sample_key_value`):
   truly keyless (no PK, no targetable UNIQUE) → leave as the documented `upsert`
   limitation, or add a PK/UNIQUE to the schema if you want them gone. **Note:**
   composite-PK junctions are *no longer* in this bucket — they convert via
   `dao.upsert(conflict=[<all PK cols>])`; sweep other `# raw` composite writes
   (e.g. remaining `mfdb_operation_artifact` / `mfdb_group_member` INSERTs) for the
   same conversion.
3. ~~**Low-value single-table SELECTs in the mixins**~~ → **DONE**: swept all 13
   `queries/*.py` files; trivial select-by-PK / equality reads now go through
   `dao.get`/`list`. What's left in `queries/` is the legitimate bespoke home
   (JOIN/aggregate/`DISTINCT`/`json_extract`/compound-`ORDER BY`/source-db/merge).

### General working rule (applies to ALL work, not just this migration)
**Every material change: update OKF, keep it traceable, commit.** The durable
statement of this lives in [`okf/workflows/change-tracking.md`](okf/workflows/change-tracking.md);
in short, for each landed unit of work:
1. Update the matching OKF concept **and** any burn-down table
   (`okf/specs/mfdb-sql-audit.md`) / PRD Definition-of-Done in the same change.
2. Append a dated bullet to `okf/log.md` — what changed, why, verification result.
3. Mark done when done (PRD `status:`/glyph, assessment row).
4. **Commit** per file or small coherent batch, message stating the change +
   verification (e.g. "suite: 420 passed") + audit delta. **Local only — never push.**

This is what makes the migration traceable: `okf/log.md` = running narrative,
the audit table = running totals, git history = small self-describing commits.

### Migration workflow that works (learned the hard way)
- **Do NOT bulk string-replace** across varied blocks — it silently corrupts (broke
  `experiments.py` in an earlier session). Use precise per-block `Edit`s.
- Convert a file → run the **full** standalone suite → fix the exact conflict/keyless
  error it reports → commit per file (or small batch). The suite is the net.
- Before converting an `INSERT OR REPLACE`/`INSERT OR IGNORE`, check the table's
  `PRAGMA index_list`/UNIQUE constraints; pick the exact `dao.upsert(conflict=[...])`
  target (or existence-check + `dao.insert` for ignore semantics).

---

## 4. Remaining lower-priority cleanup (optional; documented in memory)
- ~~Schema constraint gap on `citeulike`/`products`/`standards`/`product_categories`~~
  → **RESOLVED**: those tables **don't exist** in the reconciled schema at all (the
  "PK=NONE" reading was a false negative — `PRAGMA` on a missing table returns empty).
  Their methods were dead code and were **deleted**. Same for
  `get/add_chemical_descriptor` and `get/add_image` (live `chem_descriptors`/`images`
  columns don't match what the methods use). If you touch repository.py, watch for
  more such schema-mismatched legacy methods — verify against `PRAGMA table_info`
  before assuming a method works.
- Drop `security/base.py` `MFDBClientBase` ABC only if you also retarget ~40 type-hint
  sites to `MFDatabase` (used across `result_registry` + chisurf plugins). Decided
  **skip** this session — low value, single-backend, churny/risky.
- Delete the dead hardcoded canonical DDL text in `schema/schema.py`'s
  `CREATE_TABLES_SQL` (it's overwritten at import by `_build_permissive_ddl`, so it's
  misleading-but-harmless; don't delete the whole list — it feeds `FRESH_DB_TABLES_SQL`).
- Scaffold the **eLabFTW adapter** under `adapters/` when needed (PRD-48).

---

## 5. Pointers
- **Process rule:** `okf/workflows/change-tracking.md` (always update OKF + commit,
  keep it traceable — applies to all work).
- Architecture concept: `okf/architecture/mfdb.md` (has the "no scattered SQL is an
  antipattern" section + package layout + queries/ mixin list).
- SQL audit / burn-down: `okf/specs/mfdb-sql-audit.md`.
- Change log: `okf/log.md` (top bullets = this session's work).
- PRDs: PRD-19 (dictionary schema), PRD-26 (model-driven/dao data layer),
  PRD-24 (standalone extraction), PRD-48 (ELN adapters).
- Author/contact: Thomas-Otavio Peulen.
