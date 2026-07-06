---
type: PRD
prd: "52"
title: "PRD-52: Phasor-FLIM Imaging & Particle Tracking"
description: Add per-pixel phasor-FLIM imaging, universal-circle ROI segmentation, per-PIE/spectral-channel phasor, and phasor-based particle detection and tracking.
status: stub
phase: "unassigned"
resource: chisurf/plugins/microscopy/img_pixel_phasor/
tags: [prd, imaging]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
ChiSurf computes phasors of TCSPC decays but has no phasor-FLIM imaging: no per-pixel phasor plot, universal-semicircle navigation, ROI/threshold segmentation, per-PIE/spectral-channel phasor, or phasor-based particle detection and tracking. This PRD adds per-pixel `g,s` maps from CLSM/TTTR data with reference calibration, an interactive universal-circle plot with ROI back-projection and phasor-space unmixing, per-channel phasor, and particle detection with per-particle lifetime and trajectory linking. It reuses the existing decay-phasor math (extended to per-pixel), CLSM reconstruction tooling, and the pixel-wise FLIM plumbing pattern, with all UIs delivered as AutoForm view specs.

# Status
Stub / unassigned (STATUS TABLE authoritative). Scope and reuse identified; implementation not started.

Parent: [PRD-49](prd-49.md) (Phase 3). Related: PRD-38, PRD-40, PRD-49.

# Motivation

ChiSurf computes phasors of TCSPC *decays* but has no phasor-FLIM *imaging*: no
per-pixel phasor plot, universal semicircle navigation, ROI/threshold segmentation,
per-PIE/spectral-channel phasor, or phasor-based particle detection & tracking. The
incumbent suite's phasor, phasor-image, particle-detection, and particle-viewer apps
cover this.

# Scope

- **Per-pixel phasor image** from CLSM/TTTR data; g,s maps with reference calibration.
- **Universal-circle interactive plot** (τ markers) with circular/rectangular/threshold
  ROI back-projection to the image; multi-population linear unmixing in phasor space.
- **Per-PIE-channel and spectral phasor.**
- **Particle detection & tracking** — segment particles from phasor/FLIM, per-particle
  lifetime (τ_phase, τ_mod), Hungarian/nearest-neighbour trajectory linking.

# Reuse

- `chisurf/core/fluorescence/tcspc/phasor.py` — decay-phasor math (extend to per-pixel).
- `clsm` / tttrlib CLSM — image reconstruction, ROI/mask tooling.
- `img_pixel_mle` — pixel-wise FLIM plumbing pattern.
- skimage / scipy for segmentation + linking (or existing tracker if present).
- AutoForm + `view.json`; extend `image` section for the linked phasor↔image selection
  (add a reusable section if none fits — PRD-49 AutoForm mandate).

# Acceptance

- Headless: per-pixel phasor of a synthetic single-lifetime FLIM stack lands on the
  universal circle at the expected coordinate; ROI in phasor space selects the correct
  pixels; two-component unmixing recovers known fractions.
- Particle tracking recovers known trajectories on a simulated moving-particle stack
  ([PRD-53](prd-53.md)).

# Non-goals

Spectral unmixing as a standalone spectroscopy tool ([PRD-54](prd-54.md), may share
phasor code).

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 3).
- Shares phasor code with [PRD-55](prd-55.md) and may share spectral-unmixing code with [PRD-54](prd-54.md).
- Particle-tracking validation uses simulated stacks from [PRD-53](prd-53.md); UIs via [GUI & AutoForm](/subsystems/gui-autoform.md).
