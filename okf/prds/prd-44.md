---
type: PRD
prd: "44"
title: "PRD-44: Vendor-Neutral Dictionary Schema Namespace"
description: Renames the MFDB dictionary's local extension tags from an application-branded namespace to a store-keyed vendor-neutral one, behind a backward-compatible parser, so MFDB is usable by software beyond ChiSurf.
status: done
phase: "unassigned"
resource: overhaul/PRD-44-vendor-neutral-dictionary-schema-namespace.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-44 de-brands the MFDB dictionary's local extension tags so MFDB is a general-purpose metadata/provenance store rather than one carrying a consuming application's name in its schema authority. The schema-mapping tags (`_chisurf_schema.*`) and the branded `_chisurf_parameter` category are renamed to a namespace keyed on the store itself (`_mfdb_schema.*` / `_mfdb_parameter`), which is already the table prefix everywhere and introduces no new brand. The change is mechanical but wide (~1000 call sites) and ships behind a backward-compatible parser that dual-recognizes both spellings for one cycle, so it lands in one pass without breaking existing databases; the materialized SQL is unchanged, so no DB migration is needed. The flrCIF table `flr_chisurf_parameter` and its item ids are intentionally out of scope, since renaming those would change generated SQL.

# Status
Done (unassigned phase, STATUS TABLE authoritative). All four Definition-of-Done items met (2026-06-27): parser dual-recognizes old and new tags, the dictionary is rewritten (0 legacy / 985 new), consumers reference only the new namespace, and a back-compat test is added.

# Relationships
- A precondition for [PRD-41](prd-41.md) (dissemination): the schema namespace must be vendor-neutral to disseminate the dictionary as a product.
- Keeps the `.dic`-as-schema-authority of PRD-19 and PRD-26 brand-neutral.
- Rewrote the `mfdb_event_log` tags introduced by [PRD-43](prd-43.md) in its bulk sweep.
- Cleans up the extension namespace of the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-44-vendor-neutral-dictionary-schema-namespace.md`
