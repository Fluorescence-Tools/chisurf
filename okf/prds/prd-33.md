---
type: PRD
prd: "33"
title: "PRD-33: Acquisition-to-MFDB Registration"
description: Adds a save mode that writes a newly acquired measurement directly into MFDB with sample linkage and provenance.
status: planned
phase: "cross-cutting"
resource: overhaul/PRD-33-acquisition-mfdb-registration.md
tags: [prd, acquisition, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Offers a second acquisition save mode: register a new measurement directly into MFDB instead of, or in addition to, the file-system output folder, so measurements become first-class database objects the moment they are acquired. This path is more complex than file output because MFDB needs a provenance and ownership answer: link to an existing sample, create a new sample during acquisition, or make sample selection explicit at save time. The raw measurement is registered as an MFDB artifact preserving acquisition-settings and device metadata, while file-output mode remains available as a fallback.

# Status
Planned (Proposed).

# Relationships
- Depends on [PRD-02](prd-02.md) (sample tracking/creation), [PRD-03](prd-03.md) (result registration), and [PRD-32](prd-32.md) (output folder, which stays the default immediate path).
- Shares the "register, don't just write a file when connected" principle with [PRD-34](prd-34.md).
- Touches [MFDB (current)](/architecture/mfdb.md), acquisition/acq, and the [plugin system](/architecture/plugin-system.md).

# Source
- Primary: `overhaul/PRD-33-acquisition-mfdb-registration.md`
