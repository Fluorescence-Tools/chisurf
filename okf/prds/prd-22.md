---
type: PRD
prd: "22"
title: "PRD-22: Workflow / Pipeline Engine on the Transformer Contract"
description: Lets users compose conformant transformers into a type-checked, node-based dataflow pipeline that executes headlessly and is recorded in MFDB as a reproducible chain of operations.
status: done
phase: "4"
resource: overhaul/PRD-22-pipeline-workflow-engine.md
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-22 composes transformers into pipelines — a node-based dataflow graph at the data level — that execute and are recorded in MFDB as a chain of operations, yielding reproducible and shareable analysis workflows. A pipeline is an ordered graph of transformer invocations with nodes carrying bound parameters and edges wiring output ports to input ports, validated by port kind at definition time. A headless runner topologically evaluates the graph, dispatches each node through the replay-executor seam, and registers each step as a recorded operation with full provenance queryable through the lineage API. Definitions and runs are persisted as saveable, shareable documents.

# Status
Done (phase 4, STATUS TABLE authoritative). The headless core (Tasks 1-4) and a read-only admin pipelines viewer have landed; the visual node-based authoring editor (Task 5) is deferred.

# Relationships
- The endgame of [PRD-11](prd-11.md) and [PRD-16](prd-16.md) (operation nodes plus the transformer contract).
- Consumes [PRD-21](prd-21.md) (lineage and events, and the replay-executor seam) and [PRD-12](prd-12.md) (run status).
- The deferred visual editor is [PRD-29](prd-29.md) territory.
- Targets the [Plugins target](/specs/plugins.md) and [MFDB target](/specs/mfdb.md); registration via the [action layer](/architecture/action-layer.md).

# Source
- Primary: `overhaul/PRD-22-pipeline-workflow-engine.md`
- Supplementary: `overhaul/ORANGE3-lessons.md`
