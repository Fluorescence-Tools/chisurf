---
type: PRD
prd: "55"
title: "PRD-55: Phasor Analysis Toolkit (open-library parity)"
description: Turn ChiSurf's phasor viewer into a phasor analysis toolkit by natively implementing apparent-lifetime readout, g,s filtering, component fraction/unmixing, and cursor masks, with no new dependencies.
status: draft
phase: "unassigned"
resource: overhaul/PRD-55-phasor-analysis-toolkit.md
tags: [prd, imaging]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
ChiSurf computes calibrated per-pixel phasor `g,s` maps and renders the universal semicircle, but does nothing analytical with the phasor after plotting it — it is a phasor viewer, not a phasor analysis toolkit. Taking an open-source phasor-analysis library as a read-only reference (no import, no new dependency), this PRD natively adds four standard operations: phasor → apparent lifetime (τ_φ, τ_M), median/gaussian filtering of `g,s` maps, component fraction / linear unmixing in phasor space, and cursor/ROI masks with pseudo-color. The pure-numpy functions live in a Qt-free, tttrlib-free `analysis.py` module extending the existing phasor imaging plugin in place. Rich interactive segmentation and clustering are deliberately delegated to the companion photon-data exploration tool by handing it the per-pixel point-cloud DataFrame, rather than reimplemented.

# Status
Draft / unassigned (STATUS TABLE authoritative). Extends the existing phasor imaging plugin; constraint is zero new dependencies (numpy + scipy only).

# Relationships
- Provides the phasor math consumed and exposed over RPC by [PRD-56](prd-56.md).
- Delegates interactive gating/clustering to the companion photon-data exploration tool (reused, not reprogrammed).
- Extends a plugin in the [plugin system](/architecture/plugin-system.md); UI via [GUI & AutoForm](/subsystems/gui-autoform.md).

# Source
- Primary: `overhaul/PRD-55-phasor-analysis-toolkit.md`
