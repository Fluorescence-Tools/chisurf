---
type: PRD
prd: "35"
title: "PRD-35: Light-Path Optical Presets Stored in MFDB"
description: Migrates optical-path presets from JSON files on disk into MFDB as queryable, versioned, provenance-tracked entities.
status: planned
phase: "unassigned"
resource: overhaul/PRD-35-lightpath-mfdb-storage.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Makes every optical-path preset — the complete graph of laser lines, dichroics, splitters, filters, detector QE curves, and sample dyes — a first-class, queryable, provenance-tracked MFDB entity instead of a raw JSON file on disk. A new `mfdb_optical_preset` table stores the opaque graph payload in the content-addressed object store, with a JSON metadata column of searchable tags (laser lines, dye probes, detector count, polarizer, etc.) and RPC endpoints to list/get/put/delete presets. The graph stays opaque to SQL (the canonical node-graph serialization is unchanged); a future join table records which preset was active for a measurement, giving provenance. Migration is phased with a transparent JSON-on-disk fallback and backward compatibility retained.

# Status
Planned (Proposed). Phase 0 (JSON on disk) is done; the MFDB write path, provenance join table, and offline fallback are future phases. Complementary to, not a replacement for, the detailed instrument/optical-channel schema.

# Relationships
- Complementary to [PRD-08](prd-08.md) optical-channel/instrument schema (a preset's graph can be exported into it when applied); open question whether the use/provenance join table belongs here or with PRD-08.
- Reads probe/spectra data already in MFDB and reuses the content-addressed object store and refcounting.
- Exposes presets over RPC: [RPC target](/specs/rpc.md); stores in [MFDB (current)](/architecture/mfdb.md) / [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-35-lightpath-mfdb-storage.md`
