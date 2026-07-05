---
type: PRD
prd: "21"
title: "PRD-21: Provenance/Lineage Query API + Event Model"
description: Makes the provenance graph queryable through a first-class lineage API, stores a replayable compute spec on each derived artifact, and adds an in-process event model for reactive behaviour.
status: done
phase: "3"
resource: overhaul/PRD-21-lineage-api-event-model.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
PRD-21 replaces ad-hoc edge SQL at each call site with a first-class `Lineage` service offering ancestors, descendants, lineage-to-root, what-used / what-was-produced-from, and a provenance-graph projection for visualization. It stores on each derived artifact a compact, serializable compute spec — operation type, parameter set, and source artifact ids captured as a replayable unit — so an artifact becomes not merely traceable but recomputable, and edge rows become a derived index over these specs. A minimal in-process event bus publishes registration, operation, state-change, and calibration events post-commit and best-effort, enabling reactive workflows, audit logging, lifecycle advancement, and cache invalidation without ever breaking the registering transaction.

# Status
Done (Phase 3). The lineage read API, event bus, embedded compute spec, calibration-change impact query, and admin provenance view have landed; both reference transformers are wired with replay executors. The in-process event bus is deliberately not bridged to the networked broadcast surface.

# Relationships
- Builds on [PRD-03](prd-03.md)/[PRD-11](prd-11.md) (operations and edges) and [PRD-27](prd-27.md) (the append-only event log this API projects and publishes over).
- Serializable compute specs come from [PRD-16](prd-16.md) rule 7.
- Feeds [PRD-12](prd-12.md) (lifecycle via events) and [PRD-22](prd-22.md) (pipeline engine consumes lineage and events).
- Delivers [PRD-05](prd-05.md)'s downstream-impact promise.
- Any future networked event bridge must pass the security review in [PRD-37](prd-37.md).
- Targets the [MFDB target](/specs/mfdb.md); publishes through the [action layer](/architecture/action-layer.md).

# Source
- Primary: `overhaul/PRD-21-lineage-api-event-model.md`
- Supplementary: `overhaul/ORANGE3-lessons.md`
