---
type: PRD
prd: "19"
title: "PRD-19: Single Canonical Dictionary-Driven Schema"
description: Collapses the three overlapping table families to one canonical flrCIF-rooted schema, drives all vocabulary from the dictionary, and replaces the version-numbered migration chain with a declarative reconcile.
status: in-progress
phase: "1"
resource: overhaul/PRD-19-dict-vocab-declarative-migrations.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-19 makes flrCIF (extended via the local `.dic`) the single authoritative source for one canonical schema, its vocabularies, and its migrations. Legacy `fdb_*` tables are deleted and `mfdb_*` tables that duplicate an authoritative flrCIF concept are removed with their call sites repointed, while genuine chisurf extensions (the provenance graph, content-addressed object store, and vocabulary table) are declared as proper mmCIF extension categories foreign-keyed to flrCIF. All controlled vocabulary is seeded from dictionary enumerations, ending the duplicate-definition drift. The hand-written 39-version migration chain is replaced by a versionless declarative `reconcile_schema` that diffs the live database against the dictionary and applies the difference, reserving version stamps for one-off data backfills.

# Status
In-progress (re-verified against code 2026-07-05). The declarative engine landed: `reconcile_schema` runs on DB open, all legacy `fdb_*` tables are dropped, and vocabulary is dictionary-seeded. **Open:** the core `mfdb_*` tables are still hand-written DDL duplicated across `CREATE_TABLES_SQL` and a parallel `_CANONICAL_CHECK_SQL` (kept in sync by hand), and the `SCHEMA_VERSION = 40` stamp still lingers — so the "single canonical, dictionary-generated, no-hand-written-DDL" goal is not fully met. Corroborated by assessment [DATA-02](/specs/assessment.md#data-02) / [DATA-03](/specs/assessment.md#data-03).

# Relationships
- Fulfils [PRD-02c](prd-02c.md) (flrCIF alignment) by making flrCIF the single authoritative model.
- Subsumes the sample-table split item H3 of [PRD-25](prd-25.md).
- Foundation for [PRD-26](prd-26.md) (the model-driven layer generates over this dictionary).
- Prerequisite to [PRD-11](prd-11.md), which adds many extension categories on this canonical base.
- Benefits [PRD-24](prd-24.md) by yielding a self-contained dictionary-driven schema.
- Targets the [MFDB target](/specs/mfdb.md); see [MFDB (current)](/architecture/mfdb.md).

# Source
- Primary: `overhaul/PRD-19-dict-vocab-declarative-migrations.md`
- Supplementary: `overhaul/ORANGE3-lessons.md`
