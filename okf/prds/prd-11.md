---
type: PRD
prd: "11"
title: "PRD-11: Transformers as Abstract Data-Operation Nodes"
description: Model every data-manipulation step as a uniform MFDB operation node with typed, dictionary-declared parameters and input/output ports.
status: done
phase: "2"
resource: overhaul/PRD-11-data-operation-abstraction.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Data-manipulation steps realized as plugins (burst selection, microtime shifter, background correction, correlation) are mapped into MFDB as a single abstract data-operation node — the data-side analog of a chinet computation node — so every processing step is traceable end to end. Each transformer registers uniformly: typed data inputs and outputs (operation-artifact ports and `derived_from` edges) plus a parameter set whose schema is declared in the `.dic` dictionary (`mfdb_operation_parameter_def`) and validated on registration. The key new piece is per-operation-type parameter typing (name, type, units, bounds, description, repeatable flag), an operation-type registry linking types to their input/output kinds and schemas, and role-indexed parameter rows for variable-arity parameters — retiring bespoke per-transformer tables such as the microtime-shift table. A visual node/workflow toolkit is cited as prior art for typed I/O signals and replayable recompute rules.

# Status
Done. The uniform operation-node contract, dictionary-declared parameter schemas with validation, role-indexed parameters, and the operation-type registry are in place.

# Relationships
- Ships together with [PRD-16](prd-16.md), its plugin-side transformer-contract counterpart; together they supersede [PRD-07](prd-07.md).
- Reused by [PRD-14](prd-14.md) (protocols reference `operation_type` parameter schemas).
- Provides the provenance spine the lifecycle work ([PRD-12](prd-12.md)) hangs off.
- Subsumes the parameter side of the burst and microtime-shift ([PRD-09](prd-09.md)) work.
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-11-data-operation-abstraction.md`
- Supplementary: `overhaul/PRD-11-16-implementation-order.md`
