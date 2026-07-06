---
type: PRD
prd: "51"
title: "PRD-51: Imaging Correlation — N&B, tICS/STICS, iMSD, Spectral RICS"
description: Extend the existing RICS/CLSM core with Number & Brightness, temporal/spatiotemporal image correlation, iMSD, and crosstalk-free spectral RICS.
status: stub
phase: "unassigned"
resource: chisurf/core/models/rics/
tags: [prd, imaging, fcs]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This is the largest single cluster of methods absent versus the incumbent suite's image-correlation module. ChiSurf has RICS (roughly 2D-ICS) but lacks Number & Brightness, temporal/spatiotemporal image correlation, iMSD, and crosstalk-free spectral RICS — methods that extract oligomerization state, transport maps, and mobility from image stacks. The work reuses the existing RICS/ICS correlation core and CLSM image-from-stream tooling, and delivers every UI as an AutoForm `view.json`. Acceptance is headless: N&B recovers known brightness, iMSD recovers a known diffusion coefficient, and STICS recovers a known flow vector on simulated stacks.

# Status
Stub / unassigned (STATUS TABLE authoritative). Scope and reuse identified; implementation not started.

Parent: [PRD-49](prd-49.md) (Phase 2). Related: PRD-38, PRD-40, PRD-49.

# Motivation

The largest single ABSENT cluster vs. the incumbent suite's image-correlation module.
ChiSurf has RICS (≈2D-ICS) but lacks Number & Brightness, temporal/spatiotemporal ICS,
iMSD, and crosstalk-free spectral RICS. These extract oligomerization state, transport
maps, and mobility from image stacks — core quantitative-imaging methods.

# Scope

- **N&B** — apparent/true brightness & number from pixel intensity mean/variance;
  aggregation-state maps.
- **tICS / STICS** — temporal and spatiotemporal image correlation (flow/transport).
- **iMSD** — mean-square-displacement from image correlation: free / mobile-immobile /
  blinking variants.
- **Spectral RICS** — spectral-weighting for crosstalk-free multi-color RICS.

# Reuse

- `chisurf/core/models/rics/`, `chisurf/core/experiments/rics/ics_core.py` — correlation core.
- `clsm` / tttrlib CLSM — image-from-stream + ROI/mask handling.
- The incumbent's image-correlation model definitions — reference model forms
  (2D Gaussian, iMSD, RICS variants).
- AutoForm + `view.json` for all UIs (PRD-49 AutoForm mandate); reuse
  `image`/`waterfall`/ROI sections.

# Acceptance

- Headless: N&B recovers known brightness on a simulated fluctuating stack; iMSD recovers
  a known D; STICS recovers a known flow vector. Synthetic stacks from the
  [PRD-53](prd-53.md) simulator.
- Models render from `view.json` with no bespoke Qt; selectable in add-fit; model-editor
  headless checks pass.

# Non-goals

Phasor imaging ([PRD-52](prd-52.md)); spectral unmixing beyond RICS weighting
([PRD-54](prd-54.md)).

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 2).
- Extends the existing RICS/CLSM core; synthetic validation stacks come from [PRD-53](prd-53.md).
- All UIs via [GUI & AutoForm](/subsystems/gui-autoform.md); phasor imaging is [PRD-52](prd-52.md), spectral unmixing beyond RICS weighting is [PRD-54](prd-54.md).
