---
type: PRD
prd: "02b"
title: "PRD-02b: MFDB Admin Overhaul — Manual Inspection & Editing"
description: Make the mfdb-admin plugin inspect, add, and edit every record the sample data model produces
status: done
phase: "foundation"
resource: overhaul/PRD-02b-mfdb-admin-overhaul.md
tags: [prd, mfdb, gui]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The mfdb-admin plugin must be able to inspect, add, and edit every record the
sample data model produces — entities, multi-probe positions with flrCIF fields,
FRET pairs, optical properties and spectra, and per-sample key-value metadata —
so a user can manually verify the database is correct before downstream
workflows rely on it. The existing admin GUI predates the sample rewrite and can
only show raw table columns, so it cannot display or edit the structured data,
create samples through the definition API, or surface the full-description and
export-validation views. This PRD is the manual verification gate between the
data model and trusting it enough to wire into result-registry and pipeline
work.

# Status
Done. Serves as the human verification gate for the sample-tracking data model.

# Relationships
- Depends on [PRD-02](prd-02.md); gates downstream [PRD-03](prd-03.md) and [PRD-04](prd-04.md).
- A [plugin](/architecture/plugin-system.md) surfacing the [MFDB (current)](/architecture/mfdb.md) store; aligns with the [Plugins target](/specs/plugins.md).

# Source
- Primary: `overhaul/PRD-02b-mfdb-admin-overhaul.md`
