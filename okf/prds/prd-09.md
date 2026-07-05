---
type: PRD
prd: "09"
title: "PRD-09: Microtime Shifter — Workflow Plugin + MFDB Provenance"
description: Convert the microtime-shifter tool into a layered api/backend/cli/gui workflow plugin with RPC and full MFDB provenance.
status: done
phase: "0"
resource: overhaul/PRD-09-microtime-shifter-plugin.md
tags: [prd, plugins]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The microtime-shifter tool was a single-file Qt window with direct file read/write, no layering, and no provenance. This PRD restructures it into a workflow-ready plugin (pure `api` shift logic, `backend` RPC service, `cli`, `gui`) modeled on the burst-selection template, communicating over ZMQ JSON-RPC. Dropped TTTR files are identified and deduplicated in the MFDB object store, shifted files are written and registered as derived artifacts, and the per-channel shift values are persisted as dictionary-declared MFDB data so the operation is fully traceable. Save is connectivity-aware: with MFDB it registers into the managed object store; without it, it falls back to a file dialog and warns. Ten non-negotiable "Definition of Clean" standards govern the work (layer purity, no blobs, generated DDL, behavior-asserting tests, GUI smoke, DI over monkeypatching, idempotence).

# Status
Done. Split into api/backend/cli/gui with a versioned contract, dictionary-declared shift values, object-store dedup, and connectivity-aware save.

# Relationships
- First consumer and acceptance case for [PRD-10](prd-10.md) (dataset browser "From MFDB…").
- Its per-channel shift storage is generalized/retired into [PRD-11](prd-11.md)'s role-indexed operation parameters.
- Registration path exercised the silent-failure bug documented for [PRD-10](prd-10.md).
- Builds on the [plugin system](/architecture/plugin-system.md); target in [Plugins target](/specs/plugins.md).

# Source
- Primary: `overhaul/PRD-09-microtime-shifter-plugin.md`
