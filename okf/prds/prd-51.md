---
type: PRD
prd: "51"
title: "PRD-51: Imaging Correlation — N&B, tICS/STICS, iMSD, Spectral RICS"
description: Extend the existing RICS/CLSM core with Number & Brightness, temporal/spatiotemporal image correlation, iMSD, and crosstalk-free spectral RICS.
status: stub
phase: "unassigned"
resource: overhaul/PRD-51-imaging-correlation.md
tags: [prd, imaging, fcs]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This is the largest single cluster of methods absent versus the incumbent suite's image-correlation module. ChiSurf has RICS (roughly 2D-ICS) but lacks Number & Brightness, temporal/spatiotemporal image correlation, iMSD, and crosstalk-free spectral RICS — methods that extract oligomerization state, transport maps, and mobility from image stacks. The work reuses the existing RICS/ICS correlation core and CLSM image-from-stream tooling, and delivers every UI as an AutoForm `view.json`. Acceptance is headless: N&B recovers known brightness, iMSD recovers a known diffusion coefficient, and STICS recovers a known flow vector on simulated stacks.

# Status
Stub / unassigned (STATUS TABLE authoritative). Scope and reuse identified; implementation not started.

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 2).
- Extends the existing RICS/CLSM core; synthetic validation stacks come from [PRD-53](prd-53.md).
- All UIs via [GUI & AutoForm](/subsystems/gui-autoform.md); phasor imaging is [PRD-52](prd-52.md), spectral unmixing beyond RICS weighting is [PRD-54](prd-54.md).

# Source
- Primary: `overhaul/PRD-51-imaging-correlation.md`
