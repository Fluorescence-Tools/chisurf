---
type: PRD
prd: "08"
title: "PRD-08: Optical Configuration Schema"
description: Replace opaque setup JSON blobs with structured, queryable tables describing the full optical path from source to detector.
status: planned
phase: "4"
resource: overhaul/PRD-08-optical-configuration.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Setup configuration currently lives in opaque JSON blobs that SQL and the GUI cannot query. This PRD adds structured hardware-component tables (light sources, optical filters, dichroics, objectives, detectors, a TCSPC board) plus channel-level tables (`mfdb_optical_channel`, `mfdb_channel_setting`) modeled on the OME FilterSet/LightPath pattern, so every photon's path — which laser, filters, objective, detector, at what power/gain — becomes traceable per measurement. It includes a migration from existing blobs, `.dic` dictionary entries and enumerations, admin-GUI entity registration, and flrCIF export. The optical-path node-graph tool becomes the authoring/visualization/validation front-end and derives crosstalk and R0 as setup-level values.

# Status
Planned. Schema tables, migration helper, dictionary entries, admin registration, flrCIF export, and the node-graph front-end integration are all pending.

# Relationships
- Extends the detection-channel base tables introduced with the burst pipeline work; `mfdb_optical_channel` links to them rather than duplicating channels.
- Consumes dye spectra from [PRD-06](prd-06.md) and feeds R0/crosstalk into the calibration provenance layer.
- Independent of the operation spine ([PRD-11](prd-11.md)/[PRD-16](prd-16.md)) per the sequencing note in [PRD-11](prd-11.md).
- Builds on [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-08-optical-configuration.md`
