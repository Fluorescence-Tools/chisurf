---
type: PRD
prd: "34"
title: "PRD-34: Burst-ID Native MFDB Save and Downstream Ingest"
description: Makes MFDB the default save target for burst-identification selections when connected, and lets downstream burst tools ingest them directly from the dataset picker.
status: planned
phase: "cross-cutting"
resource: overhaul/PRD-34-bid-mfdb-native-save-ingest.md
tags: [prd, fret, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
When ChiSurf is connected to MFDB, saving a burst-identification (BID) selection should register it in MFDB (sample-linked, with the single output-folder group and burst tables) rather than only writing a loose folder, with a file-only escape hatch when disconnected. Downstream burst tools (the companion exploration tool, BVA, Burst MLE, burst-wise FCS, Trace/Burst Browser, BID→Analysis) then ingest BIDs directly by picking a registered artifact through the shared dataset picker instead of browsing the filesystem. Because burst files reference photons by index in the original TTTR file, a BID is registered as an on-disk directory reference (never copied into the object store), and downstream results are parented to the BID artifact for a clean raw → BID → analysis lineage.

# Status
Planned (Proposed).

# Relationships
- Generalizes [PRD-28](prd-28.md) (which did this for the companion exploration tool specifically) to all BID producers/consumers.
- Rides the [PRD-03](prd-03.md) / [PRD-11](prd-11.md) / [PRD-16](prd-16.md) registry/operation/transformer spine and the [PRD-10](prd-10.md) picker; identity/scope via [PRD-17](prd-17.md).
- Complements [PRD-33](prd-33.md) (acquisition→MFDB) with the same connected-save principle.
- Touches [MFDB (current)](/architecture/mfdb.md) and the [plugin system](/architecture/plugin-system.md).

# Source
- Primary: `overhaul/PRD-34-bid-mfdb-native-save-ingest.md`
