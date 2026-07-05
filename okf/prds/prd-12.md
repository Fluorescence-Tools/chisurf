---
type: PRD
prd: "12"
title: "PRD-12: Lifecycle State Machines + Transition History"
description: Turn flat entity status flags into tracked lifecycles with a recorded, validated transition log (who, when, why).
status: done
phase: "3"
resource: overhaul/PRD-12-lifecycle-state-machine.md
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
MFDB entity status was flat flags with no history or defined lifecycle. This PRD adds a single generic, dictionary-driven transition log (`mfdb_state_transition`) plus allowed-transition rules (`mfdb_state_transition_rule`), with per-entity-type state vocabularies (sample, artifact, operation). Current state is the fold over the transition log (optionally cached), giving every sample/dataset/operation an auditable "where is it and how did it get here." The repository API (`transition_state`, `get_state`, `get_state_history`) validates transitions, records operator and timestamp, and is idempotent; registration paths emit initial and advancing states best-effort. An admin lifecycle view surfaces current state and history. It is designed as a projection over the append-only event core.

# Status
Done. Schema, transition API with rule validation, registration wiring, and a standalone admin lifecycle view all landed (27 tests). The only deferred step is slotting the view into the admin tool's dock layout after the dock rewrite.

# Relationships
- A projection over the append-only event/provenance core; emits `state.changed` events.
- Operation status from [PRD-11](prd-11.md) folds into the generic transition machinery.
- Referenced by the study ([PRD-13](prd-13.md)) and protocol ([PRD-14](prd-14.md)) layers for audit.
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-12-lifecycle-state-machine.md`
