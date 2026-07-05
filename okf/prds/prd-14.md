---
type: PRD
prd: "14"
title: "PRD-14: Protocol Entity — Named, Versioned Procedures"
description: Add named, versioned measurement/processing protocols with declared parameter schemas that operations reference for reproducibility.
status: done
phase: "3"
resource: overhaul/PRD-14-protocol-entity.md
tags: [prd, mfdb, lims]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD adds a first-class **protocol**: a named, versioned procedure (measurement, processing, or analysis) with a declared parameter schema, so operations record the exact protocol and version they ran ("re-run protocol X v3 on sample Y"). `mfdb_protocol` links a protocol to its `operation_type` (reusing PRD-11's operation-parameter schemas rather than forking a stack) and, for measurements, to a setup. Protocols are append-only by version — editing creates a new version while operations keep the version they ran — and `mfdb_operation` gains `protocol_id`/`protocol_version` columns. Registration validates operation parameters against the protocol/operation schema and requires the operation type to match. Repository CRUD with versioning, scoping, RPC, and a standalone admin protocols view are provided.

# Status
Done. Schema, versioned append-only CRUD, operation references with validation, and an admin view landed (43 tests). Wiring the view into the admin dock layout is deferred to the dock rewrite.

# Relationships
- Builds on [PRD-11](prd-11.md) (operation nodes + parameter schemas) and the setup/calibration versioning work.
- Provides the "how was it measured/processed" record referenced by the lifecycle ([PRD-12](prd-12.md)) and study ([PRD-13](prd-13.md)) layers.
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-14-protocol-entity.md`
