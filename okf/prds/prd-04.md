---
type: PRD
prd: "04"
title: "PRD-04: Stable Burst Pipeline MFDB Integration"
description: Register burst-selection results in MFDB with stable, queryable provenance across all callers
status: in-progress
phase: "0"
resource: overhaul/PRD-04-burst-pipeline.md
tags: [prd, mfdb, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
When the Burst Selection plugin produces burst tables, those results are
registered in MFDB with stable, queryable provenance (raw TTTR artifacts →
burst_selection operation → burst_table artifacts → optional sidecar outputs),
identically across GUI, CLI, RPC, and direct API callers. The scientific
analysis API stays the source of truth for computing burst results; MFDB
registration is a narrow archival side effect layered on top. It also makes the
acquisition setup — detector definitions plus PIE/microtime window (channel)
definitions and the TTTR reading routine — a first-class MFDB record so a burst
run can be traced back to what setup produced the photons.

# Status
In progress. Burst Selection is the reference implementation for
workflow-ready plugins with MFDB archival; it must not reimplement storage,
operation/parameter recording, or payload serialization owned by the registry
and codec layers.

# Relationships
- Builds on [PRD-030](prd-030.md) codecs and the [PRD-03](prd-03.md) registry; adds setup/channel records extended later by an optical-configuration PRD.
- Reference archival path over the [MFDB (current)](/architecture/mfdb.md) store for a [plugin](/architecture/plugin-system.md); relates to [Plugins target](/specs/plugins.md).

# Source
- Primary: `overhaul/PRD-04-burst-pipeline.md`
- Supplementary: `overhaul/PRD-04-code-review.md`, `overhaul/PRD-04-code-review-followup.md`, `overhaul/PRD-04-code-review-sidequest.md`, `overhaul/PRD-04-code-review-sidequest-B.md`, `overhaul/PRD-04-code-review-sidequest-C.md`, `overhaul/PRD-04-code-review-sidequest-D.md`, `overhaul/PRD-04-code-review-addendum2.md`, `overhaul/PRD-04-punch-list.md`, `overhaul/PRD-04-sidequest-punch-list.md`
