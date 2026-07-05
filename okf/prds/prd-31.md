---
type: PRD
prd: "31"
title: "PRD-31: Headless CLI for the Companion Photon-Data Exploration Tool"
description: Adds a windowless CLI to the companion exploration tool for parameter-based burst filtering and imaging, integrated with MFDB.
status: planned
phase: "2"
resource: overhaul/PRD-31-ndxplorer-headless-cli.md
tags: [prd, imaging, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Gives the companion photon-data exploration tool a headless CLI for its two core jobs, both previously GUI-only: burst filtering (select a subset of bursts by parameter ranges/gates and emit a filtered burst selection) and imaging (render intensity or per-pixel parameter maps from image-axis/CLSM data, apply gates/ROIs, and export images or a masked sub-selection). Both run with no window, print JSON to stdout, and complete the MFDB round trip. Pure primitives live in the chisurf-free external module; MFDB resolution and write-back live in the ChiSurf-side wrapper. Filtered/masked outputs stay a reference beside the same TTTR so photon-index linkage is preserved.

# Status
Planned overall, though the header notes the two primitives and both wrappers landed 2026-06-27; the `image` MFDB round trip is implemented but not yet covered by a ChiSurf-side test.

# Relationships
- Completes the CLI leg of [PRD-28](prd-28.md) (companion-tool ↔ MFDB burst integration).
- Complements [PRD-30](prd-30.md) (Unix-pipe CLI composability) by making the exploration tool a headless filter/imaging stage.
- Honors [PRD-23](prd-23.md) (thin widgets) and [PRD-17](prd-17.md) (identity) for MFDB write-back.
- Registers results via [PRD-03](prd-03.md) result registry; resolves through [MFDB (current)](/architecture/mfdb.md) and the [plugin system](/architecture/plugin-system.md).

# Source
- Primary: `overhaul/PRD-31-ndxplorer-headless-cli.md`
- Supplementary: `overhaul/PRD-31-ndxplorer-pyqtgraph-migration.md` — a separate, completed (2026-06-24) GUI-migration variant carrying the same number, which replaced a legacy Qt plotting toolkit and most matplotlib usage in the exploration tool's GUI with pyqtgraph (single visualization stack, faster startup); that migration is done.
