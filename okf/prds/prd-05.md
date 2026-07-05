---
type: PRD
prd: "05"
title: "PRD-05: Calibration Provenance"
description: Track calibration parameters in MFDB with links to the reference measurements they derive from
status: in-progress
phase: "4"
resource: overhaul/PRD-05-calibration-provenance.md
tags: [prd, mfdb, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Calibration parameters — g-factor, gamma, crosstalk, direct excitation, donor
lifetime, Förster radius — are tracked in MFDB with links to the reference
measurements they were derived from, so that when a calibration value changes,
every downstream fit that used it can be identified. Today these live as free
fitting-parameter objects with no record of where their values came from. The
PRD adds a calibration data model and provenance edges, and allows recording
values that are predicted from a setup's structured optical path with a
distinct "computed-from-optics" provenance source, so predicted and measured
crosstalk/R₀ can be compared.

# Status
In progress. Depends on the result registry for storing calibration records and
wiring provenance edges; some values can be sourced from an optical-configuration
setup rather than a measurement.

# Relationships
- Depends on the [PRD-03](prd-03.md) result registry; consumes samples/FRET pairs from [PRD-02](prd-02.md).
- Predicted values relate to an optical-configuration setup extended from [PRD-04](prd-04.md).
- Records calibration provenance in the [MFDB (current)](/architecture/mfdb.md) store toward the [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-05-calibration-provenance.md`
