# START HERE — MFDB / chisurf overhaul: current state & next steps

_Last updated: 2026-06-22. Branch: `development` (work stays on dev; do **not** merge
to `master`). Authoritative plan: `MASTER-ORDER.md`._

## Where we are

We are in **Phase 1 (architecture foundations)** of `MASTER-ORDER.md`. The
PRD-19 **groundwork** just landed and is green; the rest of Phase 1 is still open.

### Just committed (on `development`)

- `447e481e` — **mfdb: land PRD-19 schema groundwork + boundary link validation**
  - `reconcile_schema` (declarative migrate toward the dictionary).
  - `OPERATION_TYPES` now derived from `.dic` enumerations (vocab-from-dictionary).
  - `flr_sample` is the canonical sample-name source; payload/base split
    (`payload_codec.py`, `payload_models.py`, `base.py`); obsolete `pipeline.py` removed.
  - **PRD-25 N3 / option (b):** new `LinkValidationError` raised *before* the write
    transaction (no partial artifact/object rows). The burst MFDB pipeline catches it
    and reports a bad `sample_id` as a warning instead of aborting; genuine write
    errors still fail loud.
- `50c22d6c` — **Document flr_sample canonical-name change; assert details persistence**
  - `flr_sample.description` = display name; long text persisted in `flr_sample.details`
    (DB-level assertion added). Breaking change recorded in `CHANGELOG.md`.

### Test status

- mfdb core suite **green**: 44 passed across `test_schema_from_dictionary`,
  `test_setup_prerequisites`, `test_sample_manager`, `test_burst_pipeline_mfdb`.
- **Known pre-existing failure (NOT ours, unrelated):**
  `test/plugins/test_sample_database_plugin.py::test_gui_starts_embedded_mfdb_rpc_when_unavailable`
  — fails at HEAD too (GUI embedded-RPC bootstrap). Also two collection errors in
  `test_becker_hickl_set.py` / `test_bhfiles.py` (`ImportError: BeckerHicklSetReader`)
  from a *different* dirty subsystem.
- ⚠️ Batching GUI (`test_sample_database_plugin.py`) with other files under
  `QT_QPA_PLATFORM=offscreen` can **hang**. Run GUI files separately. (Exactly what
  PRD-18's hermetic harness is meant to fix.)

### Working-tree reality (important)

- The repo has a **large uncommitted working tree** (~200 files across plugins, gui,
  etc.) — the broader session's WIP. The `core/mfdb` subsystem is now fully committed
  and clean; everything else remains dirty and is **out of scope** for the PRD-19 work.
- `development` is well ahead of `origin/development` (unpushed) and ~1944 commits
  ahead of `master`. **No push, no master merge** unless explicitly asked.

## PRD-19 decision: option B (locked)

MFDB is unreleased and there is no real data yet → **pre-PRD-19 databases are
disposable**. No forward migration is required; an incompatible old DB is deleted
and recreated. This is why the version chain could be removed wholesale.

## What PRD-19 still needs (its DoD is not fully met)

Progress (committed): vocab is fully `.dic`-sourced; the fresh-DB path is already
legacy-free (`FRESH_DB_TABLES_SQL` filters `fdb_*`; no `mfdb_sample` CREATE); **the
39-step migration waterfall is removed** (`bb837a07`, −1862 lines) — fresh + existing
DBs go through `CREATE IF NOT EXISTS` + `reconcile_schema`. Remaining:

1. ~~**Drop the version chain (idea K).**~~ **DONE** (`bb837a07`); `SCHEMA_VERSION` is
   now only a harmless stamp.
2. **Delete dead legacy code:** `fdb_*` CREATEs still sit (unused) in the raw
   `CREATE_TABLES_SQL` + filter; `graph.py` `fdb_provenance_edge` reads; many
   now-unused migration helpers (`_ensure_column`, etc.).
3. **Collapse duplicates onto flrCIF:** remove `mfdb_*` tables that duplicate a flrCIF
   concept; repoint reads/writes to the authoritative `flr_*`. **flrCIF is
   authoritative and the `.dic` extends it** — do NOT demote flrCIF to an export codec.
4. **Declare genuine extensions** (provenance graph, object store, vocab) as proper
   flrCIF extension categories in `mfdb_flr_ext.dic`, FK'd to flrCIF.
5. **Extend `validate_mapping` gate** to assert live ⊇ declared + vocab == dictionary
   + no legacy/duplicate tables after reconcile. (Vocab duplication already removed.)

## Next steps (in order, per MASTER-ORDER)

1. **Finish PRD-19** (remaining items above: dead-code deletion, gate extension) —
   minimizes rework before the spine. Version-chain removal is done.
2. **PRD-18** — DI + hermetic temp-DB test harness. Flagged "land immediately":
   fixes the real-DB pollution and the Qt-batching hangs we keep hitting.
3. **PRD-17** — one canonical identity/session context (owner-mismatch class).
4. **PRD-27** — append-only provenance/state core (decision gates PRD-12/21).
5. Then **Phase 2:** PRD-11 + PRD-16 spine, refactor burst + microtime-shifter as the
   two reference transformers, then **PRD-26** (model-driven layer). Apply PRD-23
   (thin widgets) per tool as touched.

## Quick reference

- Run mfdb tests (avoid GUI batching hangs):
  ```
  QT_QPA_PLATFORM=offscreen PYTHONPATH="modules/chinet:modules/imp-tricks/src:." \
    /Users/tpeulen/mambaforge/envs/arm64/bin/python3 -m pytest -p no:cov -o addopts='' \
    test/fio/test_schema_from_dictionary.py test/fio/test_setup_prerequisites.py \
    test/fio/test_sample_manager.py test/fio/test_burst_pipeline_mfdb.py
  ```
- Real user DB (do **not** let tests touch it): `~/.chisurf/flr/sample_management.db`.
- Plan docs: `MASTER-ORDER.md` (order), `README.md` (index), `MFDB-architecture-ideas.md`
  (rationale A–H + bold I–N: I/K→PRD-19, J→PRD-26, M→PRD-27, N→PRD-25).
