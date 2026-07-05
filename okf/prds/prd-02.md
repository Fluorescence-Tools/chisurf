---
type: PRD
prd: "02"
title: "PRD-02: Sample Tracking — Deep Sample Description"
description: Link every dataset and result to a full atomistic, flrCIF-aligned sample description
status: done
phase: "foundation"
resource: overhaul/PRD-02-sample-tracking.md
tags: [prd, mfdb, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Every dataset and analysis result in MFDB is linked to a sample, where a sample
is a full atomistic description — the biomolecule and its sequence, labeling
positions, fluorescent probes and their photophysical properties, FRET pairs,
and buffer conditions — modeled as a graph across the flr_* tables and
exportable as valid flrCIF. It restructures `SampleDefinition` into typed
entity, probe, and FRET-pair sub-models (probes carry no intrinsic
donor/acceptor role; that is pair-relative), auto-populates spectra for known
dyes, and makes `create_sample` populate the full flrCIF data model plus a
lightweight index row. Supports 2/3/4-color and homo-FRET, single-label FCS, and
multi-chain complexes.

# Status
Done. A companion GUI verification gate ([PRD-02b](prd-02b.md)) exists so the
structured data can be inspected and confirmed before downstream wiring.

# Relationships
- Depends on [PRD-020](prd-020.md) (ORM boundary) and [PRD-02a](prd-02a.md) (dictionary validation).
- GUI counterpart / verification gate: [PRD-02b](prd-02b.md); export alignment: [PRD-02c](prd-02c.md).
- Populates sample tables in the [MFDB (current)](/architecture/mfdb.md) store toward the [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-02-sample-tracking.md`
- Supplementary: `overhaul/PRD-02_COMPLETION_REPORT.md`, `overhaul/PRD-02-IMPLEMENTATION_SUMMARY.md`, `overhaul/CODE_REVIEW_PRD-02.md`
