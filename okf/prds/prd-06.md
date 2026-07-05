---
type: PRD
prd: "06"
title: "PRD-06: Expand the Fluorophore Database"
description: Populate MFDB with real, provenance-tracked spectral data for common dyes and compute Förster radii from spectral overlap.
status: in-progress
phase: "4"
resource: overhaul/PRD-06-fluorophore-database.md
tags: [prd, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
MFDB should ship real spectral data (absorption/emission, quantum yield, extinction) for common single-molecule FRET dyes, so a donor–acceptor pair yields a Förster radius computed from the spectral-overlap integral. The core Förster calculator (`forster.py`: overlap integral, R0, R0-from-spectra) has landed. The remaining work integrates two previously un-shipped internal tools — a fluorophore-curation engine and a spectral scraper carrying ~1,954 scraped probes — into the live MFDB as a single store, adds `source`/verification/quality provenance, and routes downstream consumers (R0 lookup, calibration feeds) to approved-only data. An AI-assisted triage pass (provider-neutral, via the existing local/OpenAI-compatible AI settings) proposes categories, quality grades, name canonicalization, and deduplication for human approval, never auto-approving.

# Status
In progress. Task 1 (Förster calculator + tests) is done; integration of the curation/scraper tools, the verification/approval workflow, and AI triage remain.

# Relationships
- Feeds R0 / crosstalk into [PRD-08](prd-08.md) (optical configuration) and the calibration provenance work.
- Reagent inventory [PRD-15](prd-15.md) may link fluorophore lots to probe records.
- Builds on the dictionary-driven schema of [MFDB (current)](/architecture/mfdb.md); target in [MFDB target](/specs/mfdb.md).

# Source
- Primary: `overhaul/PRD-06-fluorophore-database.md`
