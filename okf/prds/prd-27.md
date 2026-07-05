---
type: PRD
prd: "27"
title: "PRD-27: Event-Sourced, Append-Only Provenance Core"
description: Locks the append-only-lite provenance direction and implements branch/event-log pieces while full reconstructable append-only provenance remains incomplete.
status: in-progress
phase: "1"
resource: overhaul/PRD-27-event-sourced-provenance-core.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Because MFDB is unreleased, this PRD decides the provenance-core shape early and then incrementally hardens it: recorded facts should be appended rather than destructively rewritten, "delete" should become a tombstone, and current state/lifecycle should be projections folded over append-only records. A go/no-go chose append-only-lite (keep the current tables, make provenance/state changes append-only where practical) over full event-sourcing, keeping the latter as the documented end state. Branches are represented as named pointers to operation heads, enabling limited "what-if" reprocessing without touching the main line.

# Status
In-progress (re-verified against code 2026-07-05). The append-only-lite decision is locked, lifecycle transitions are append-only, an event-log table exists for GUI history, and branch/head APIs plus branching tests are present. **Open:** the PRD's implementation DoD is broader than the code: full provenance/state reconstructability is not proven, several core tables still use `UPDATE`/upsert paths for recorded facts, and full single-log event-sourcing remains only the documented future end state.

# Relationships
- Reframes and must precede [PRD-12](prd-12.md) (lifecycle = state-log projection) and [PRD-21](prd-21.md) (events = views over the same record).
- Sits under the operation spine of [PRD-11](prd-11.md) / [PRD-16](prd-16.md); operations append events.
- Branching enables the "what-if" replay used by [PRD-29](prd-29.md).
- Foundation for the metadata/provenance store: [MFDB (current)](/architecture/mfdb.md), [MFDB target](/specs/mfdb.md).
- Replayable compute-spec idea drawn from a visual node/workflow toolkit's prior art.

# Source
- Primary: `overhaul/PRD-27-event-sourced-provenance-core.md`
