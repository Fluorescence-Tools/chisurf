---
type: PRD
prd: "43"
title: "PRD-43: Align GUI Operation History with MFDB Provenance"
description: Makes the in-memory operation history a projection over a durable MFDB event log so undo/redo and the exploration trail survive database save/restore, and incrementally aligns the GUI event stream with the backend provenance model.
status: in-progress
phase: "cross-cutting"
resource: overhaul/PRD-43-history-mfdb-alignment.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-43 closes the gap where saving a project to a local `.csp` archive preserved the full action history but saving to MFDB discarded it, losing the undo stack and scientific-exploration trail. The original "opaque history blob" plan was superseded by a stronger design in which MFDB is the durable source of truth and the in-memory `OperationHistory` is a projection over it, delivered in four staged headless-tested increments (Qt-free replay, dictionary-declared vocabulary alignment, bounded snapshots, and an append-only `mfdb_event_log` with best-effort dual-write). Making recording live again also exposed and fixed three stacked latent bugs in the interactive undo/redo replay path plus UID-remapping on redo. Later phases (auto-creating operation rows in the dispatcher, transactional GUI-to-DB operations, unified replay) ride other architecture PRDs.

# Status
In-progress (cross-cutting phase, STATUS TABLE authoritative). Recording, undo, and redo (re-create fits, restore name-keyed state, remap model-state fit groups) work; several known undo/redo edges and duplicate-recording consolidation remain, and Phases 2–4 are deferred.

# Relationships
- Extends PRD-01 (fix roundtrip): closes the remaining MFDB round-trip history gap.
- Uses the PRD-03 object store and artifact-registration infrastructure.
- Aligns GUI events with the PRD-21 lineage API / event model and folds into the PRD-27 append-only provenance core in later phases.
- Introduced the `mfdb_event_log` table that [PRD-44](prd-44.md)'s namespace sweep rewrites.
- Projects the GUI history over the [MFDB (current)](/architecture/mfdb.md) toward the [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-43-history-mfdb-alignment.md`
