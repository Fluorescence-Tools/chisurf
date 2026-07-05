---
type: PRD
prd: "01"
title: "PRD-01: Fix MFDB Project Round-Trip"
description: Make archiving a project to MFDB and restoring it produce an identical project
status: planned
phase: "0"
resource: overhaul/PRD-01-fix-roundtrip.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Archiving a project to MFDB and then restoring it must reproduce the original
project exactly. Today the round-trip is lossy: only the inner fit-state payload
is stored (dropping fit id, name, model name, plot state, and range), multiple
datasets collapse to a single key, project-level metadata is not persisted, and
chinet sessions, parameters, and dependency edges are discarded on restore. This
PRD stores full fit records plus project metadata during archival and reads back
fit groups, all datasets, parameters, and edges on restore, backed by a
round-trip regression test.

# Status
Planned. The PRD enumerates concrete fixes in `project_archiver.py` and the
project-browser restore handler, with a new round-trip test as the gate.

# Relationships
- Foundational fix that later result/provenance PRDs build on: [PRD-030](prd-030.md), [PRD-03](prd-03.md).
- Operates on the [MFDB (current)](/architecture/mfdb.md) store toward its [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-01-fix-roundtrip.md`
