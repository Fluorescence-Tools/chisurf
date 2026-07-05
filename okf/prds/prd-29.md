---
type: PRD
prd: "29"
title: "PRD-29: Visual Burst Programming — Node-Graph Editor for Burst Analysis"
description: Turns the existing node editor into a visual programming canvas for composing, running, previewing, and provenance-recording burst-analysis pipelines.
status: planned
phase: "2"
resource: overhaul/PRD-29-visual-burst-programming.md
tags: [prd, fret, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Turns the existing node editor into a visual programming language for burst analysis: a canvas where users graphically compose pipelines (load TTTR → filter → select bursts → compute FRET → fit decays → export). Each node is a typed, executable operation following the transformer contract; the graph runs live via the reactive chinet runtime with intermediate previews. Every execution is recorded in MFDB with full provenance (operations, artifacts, parameters, edges), and pipelines are saved as MFDB artifacts that reference data by UUID so they are shareable and re-executable, including headlessly. Re-execution with changed parameters creates a branch rather than mutating the original.

# Status
Planned. Merges three existing pieces (node editor, chinet runtime, MFDB); phased plan covers burst node types, provenance recording, live interactivity, and polish including headless execution.

# Relationships
- Delivers the visual counterpart to [PRD-22](prd-22.md) (pipeline engine), sharing the headless runner and transformer contract.
- Nodes map to operations and typed parameters per [PRD-11](prd-11.md) and follow the [PRD-16](prd-16.md) transformer contract.
- Stores outputs via [PRD-03](prd-03.md) result registry; edges queryable via [PRD-21](prd-21.md) lineage API; versioning/branching via [PRD-27](prd-27.md).
- The `burst_select` node wraps the [PRD-04](prd-04.md) burst pipeline.
- Uses [MFDB (current)](/architecture/mfdb.md); modeled on a visual node-graph editor's interaction style.

# Source
- Primary: `overhaul/PRD-29-visual-burst-programming.md`
