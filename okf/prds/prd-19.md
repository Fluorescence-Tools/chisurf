---
type: PRD
prd: "19"
title: "PRD-19: Single Canonical Dictionary-Driven Schema"
description: Collapses the three overlapping table families to one canonical flrCIF-rooted schema, drives all vocabulary from the dictionary, and replaces the version-numbered migration chain with a declarative reconcile.
status: in-progress
phase: "1"
resource: chisurf/core/mfdb/
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-19 makes flrCIF (extended via the local `.dic`) the single authoritative source for one canonical schema, its vocabularies, and its migrations. Legacy `fdb_*` tables are deleted and `mfdb_*` tables that duplicate an authoritative flrCIF concept are removed with their call sites repointed, while genuine chisurf extensions (the provenance graph, content-addressed object store, and vocabulary table) are declared as proper mmCIF extension categories foreign-keyed to flrCIF. All controlled vocabulary is seeded from dictionary enumerations, ending the duplicate-definition drift. The hand-written 39-version migration chain is replaced by a versionless declarative `reconcile_schema` that diffs the live database against the dictionary and applies the difference, reserving version stamps for one-off data backfills.

# Status
In-progress (re-verified against code 2026-07-05). The declarative engine landed: `reconcile_schema` runs on DB open, all legacy `fdb_*` tables are dropped, and vocabulary is dictionary-seeded. **Open:** the core `mfdb_*` tables are still hand-written DDL duplicated across `CREATE_TABLES_SQL` and a parallel `_CANONICAL_CHECK_SQL` (kept in sync by hand), and the `SCHEMA_VERSION = 40` stamp still lingers — so the "single canonical, dictionary-generated, no-hand-written-DDL" goal is not fully met. Corroborated by assessment [DATA-02](/specs/assessment.md#data-02) / [DATA-03](/specs/assessment.md#data-03).

# Scope note
This folds in two related ideas: collapsing the three table families to one canonical model, and a versionless declarative schema. MFDB is unreleased — do the breaking consolidation now, before more schema lands.

**Authority rule (read first): flrCIF/pdbx is the authoritative schema. We do not replace it or demote it to an export format — we *extend* it.** The chisurf extension dictionary (`mfdb_flr_ext.dic`) adds, in proper mmCIF extension form, only what flrCIF/IHM/pdbx does not already cover (chisurf's provenance graph and object store). Where `mfdb_*` today *duplicates* a flrCIF concept, the flrCIF table wins and the duplicate is removed.

# Goal
Make flrCIF (extended via the `.dic`) the single source of truth for **one** canonical schema, its **vocabularies**, and migrations:

1. **Collapse to one family rooted in flrCIF.** Delete legacy `fdb_*`. Remove the `mfdb_*` tables that duplicate flrCIF concepts (sample, experiment, measurement, instrument/setup, …) and read/write the authoritative `flr_*` tables instead.
2. **Keep genuine extensions, declared as flrCIF extensions.** The provenance DAG (operation / artifact / edge / parameter), the content-addressed object store, and the vocabulary table have no flrCIF equivalent — keep them, but declare them as proper extension categories in `mfdb_flr_ext.dic`, aligned with flrCIF/pdbx naming and FK'd to the flrCIF entities they describe. **Prefer an existing flrCIF/IHM/pdbx category over inventing one** wherever the concept already exists (e.g. IHM data-transformation / software / provenance categories).
3. **Drive all vocab from the dictionary.**
4. **Replace the hand-written version-numbered migration chain with a declarative reconcile to the dictionary.**

# Evidence (why)
- **Three table families** — legacy `fdb_*`, authoritative flrCIF `flr_*`, and the `mfdb_*` layer — with reads/writes split across them. The split, specifically `mfdb_*` **duplicating** an authoritative flrCIF table, is the *root* of the sample-name bug (`mfdb_sample.display_name` shadowed `flr_sample.description`) and the `create_sample` dual-write. The fix is "flrCIF is authoritative; drop the duplicate", not "introduce a fourth canonical model."
- **Vocabulary defined twice** — `OPERATION_TYPES` (models.py) and the `bootstrap_vocabulary` dict (schema.py); `microtime_shift` added to one, validated against the other → silent registration failure.
- **39 linear migration versions** with scattered `_ensure_column` and local-import hacks to dodge `UnboundLocalError` (the v39 scoping bug). Brittle.

# Design

## 1. flrCIF authoritative; one family; extensions declared in the `.dic`
- **flrCIF `flr_*` is the live, authoritative schema** for everything flrCIF covers. Legacy `fdb_*` is deleted. `mfdb_*` tables that duplicate a flrCIF concept are removed and their call sites repointed at the authoritative `flr_*` table.
- **Genuine extensions stay** (chisurf provenance graph + object store + vocab) but are **declared as mmCIF extension categories** in `mfdb_flr_ext.dic`, following flrCIF/pdbx conventions, with `_chisurf_schema` bridges and foreign keys into the flrCIF entities. Before adding any extension category, check flrCIF/IHM/pdbx for an existing category that already models the concept and extend/use that instead.
- Every `_chisurf_schema` bridge therefore maps each dictionary item to exactly one live column — either an authoritative `flr_*` column or a declared extension column. No `fdb_*` targets; no two tables modelling the same concept. Removes the dual-table read-split bug class and the dual-write (subsumes [PRD-25](prd-25.md) H3; this is the fulfilment of [PRD-02c](prd-02c.md)'s flrCIF alignment, not its reversal).

## 2. Vocabulary from the `.dic`
- Operation types, artifact kinds, relationship types, statuses, state vocabularies are dictionary enumerations (`_item_enumeration.value`) on the relevant flrCIF or extension items. Seed `mfdb_vocabulary` from the `.dic`; delete `OPERATION_TYPES` and the bootstrap dict. One authored source; no drift.

## 3. Versionless declarative reconcile
- The generator knows the target schema from the dictionary (authoritative flrCIF categories + declared extensions); `introspect_sqlite_schema` knows the live DB. `reconcile_schema(conn, dictionary)` computes the **diff** (missing tables → `CREATE`; missing columns → `ALTER ADD`; missing indexes) and applies it. The DB is *made to match the dictionary*.
- **Drop the structural version chain.** Unreleased ⇒ reset to a single baseline + reconcile on open; keep only a tiny set of one-off **data** backfills (run-once, recorded). No more `if version < N` structural blocks, no local-import hacks.
- **Prior art (an established node-graph data-analysis toolkit).** That toolkit never hand-writes structural DDL migrations — it keeps structure implicit and uses a settings-version stamp plus a `migrate_settings(settings, version)` step to migrate only *stored values/params*. This confirms the split here: structure is declarative (`reconcile_schema`), and version stamps are reserved for **data** backfills only. Its explicit domain-conversion object (which adapts records from one typed schema to another) is the pattern to follow for the flrCIF export/import codec and any data backfill — an explicit conversion, not ad-hoc SQL.
- Gate: `validate_mapping` extended to assert live ⊇ declared (no missing declared column) and vocab == dictionary.

# Tasks
1. Repoint every `_chisurf_schema` bridge to its authoritative live column: a `flr_*` column where flrCIF covers the concept, else a declared extension column. For each `mfdb_*` table, classify it as **duplicate** (remove; repoint to `flr_*`) or **genuine extension** (declare in `mfdb_flr_ext.dic` as a flrCIF extension, FK'd to flrCIF). Delete legacy `fdb_*`. Migrate existing data once.
2. Audit every `flr_`/`fdb_`/`mfdb_` reference in `chisurf/core/mfdb/*.py` (repository, orm/sample_repository, importer, graph, object_store, chinet_adapter, project_archiver, api) and route reads/writes to the authoritative table.
3. Vocab seeding from the `.dic`; remove Python vocab duplication.
4. `reconcile_schema` (diff + apply); route fresh-DB and open paths through it; baseline reset; reserve version stamps for data backfills only.
5. Total-coverage tests: a freshly reconciled DB matches the dictionary exactly (tables/columns/indexes/vocab); no `fdb_*` tables and no duplicate-of-flrCIF `mfdb_*` tables remain; the sample-name round-trip reads `flr_sample` only.

# Definition of Done
- [ ] flrCIF `flr_*` is the authoritative live schema; legacy `fdb_*` removed; `mfdb_*` duplicates of flrCIF concepts removed and call sites repointed.
- [ ] Remaining chisurf extensions are declared in `mfdb_flr_ext.dic` as proper flrCIF extension categories, FK'd to flrCIF; no concept is modelled twice.
- [ ] All vocab seeded from the `.dic`; no Python vocab duplication.
- [ ] `reconcile_schema` brings any DB to the dictionary schema; structural version chain gone; new schema = a `.dic` edit only.
- [ ] Gate asserts live ⊇ declared + vocab match.

# Definition of Clean
flrCIF is authoritative and the `.dic` extends it (no parallel/duplicate model, no demotion of flrCIF to an export format); the `.dic` dictates schema + vocab + migrations; one live source per concept; declarative over hand-written migrations; delete legacy, don't alias; behavior-asserting total-coverage tests.

# Relationships
- Fulfils [PRD-02c](prd-02c.md) (flrCIF alignment) by making flrCIF the single authoritative model.
- Subsumes the sample-table split item H3 of [PRD-25](prd-25.md).
- Foundation for [PRD-26](prd-26.md) (the model-driven layer generates over this dictionary).
- Prerequisite to [PRD-11](prd-11.md), which adds many extension categories on this canonical base.
- Benefits [PRD-24](prd-24.md) by yielding a self-contained dictionary-driven schema.
- Targets the [MFDB target](/specs/mfdb.md); see [MFDB (current)](/architecture/mfdb.md).
