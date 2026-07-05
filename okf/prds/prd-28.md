---
type: PRD
prd: "28"
title: "PRD-28: Companion-Tool ↔ MFDB Burst-Selection Round Trip"
description: Opens registered burst selections from MFDB in the companion exploration tool; direct send-current-result from Burst Selection remains open.
status: in-progress
phase: "2"
resource: overhaul/PRD-28-ndxplorer-burst-integration.md
tags: [prd, fret, mfdb]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Closes the loop so a burst selection can move between the Burst Selection tool, MFDB, and the companion photon-data exploration tool via the existing sample/measurement picker, giving a hands-on test of the Phase-2 operation/transformer spine. The implemented direction opens a registered burst selection from MFDB into the exploration tool; the reverse "send the just-produced selection" workflow remains unfinished. Because burst files reference photons by index in the original TTTR file, the selection is registered as an on-disk directory reference (not copied into the object store), and multi-file runs appear as a single group. GUI actions only orchestrate pick → resolve path → hand off; the external exploration module stays free of ChiSurf imports.

# Status
In-progress (re-verified against code 2026-07-05). Landed: open-from-MFDB, the path-resolution helper, the CLI handoff with `--mfdb` registration, and path-resolution tests. **Open:** the Burst Selection GUI action still calls `open_burst_selection_from_mfdb()` (picker-driven open-from-MFDB) rather than sending the current result path via `send_path_to_ndxplorer()`, so direction B and the original two-way DoD are not complete.

# Relationships
- Manual-test enabler for [PRD-04](prd-04.md) (burst pipeline) and the Phase-2 spine ([PRD-11](prd-11.md) / [PRD-16](prd-16.md)).
- Reuses [PRD-10](prd-10.md) (dataset browser / picker) and [PRD-17](prd-17.md) (identity resolver); applies [PRD-23](prd-23.md) (thin widgets).
- Headless exploration-tool leg specified separately in [PRD-31](prd-31.md); generalized to all burst-ID producers/consumers in [PRD-34](prd-34.md).
- Touches [MFDB (current)](/architecture/mfdb.md) and the [plugin system](/architecture/plugin-system.md).

# Source
- Primary: `overhaul/PRD-28-ndxplorer-burst-integration.md`
