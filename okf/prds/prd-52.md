---
type: PRD
prd: "52"
title: "PRD-52: Phasor-FLIM Imaging & Particle Tracking"
description: Add per-pixel phasor-FLIM imaging, universal-circle ROI segmentation, per-PIE/spectral-channel phasor, and phasor-based particle detection and tracking.
status: stub
phase: "unassigned"
resource: overhaul/PRD-52-phasor-flim-imaging.md
tags: [prd, imaging]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
ChiSurf computes phasors of TCSPC decays but has no phasor-FLIM imaging: no per-pixel phasor plot, universal-semicircle navigation, ROI/threshold segmentation, per-PIE/spectral-channel phasor, or phasor-based particle detection and tracking. This PRD adds per-pixel `g,s` maps from CLSM/TTTR data with reference calibration, an interactive universal-circle plot with ROI back-projection and phasor-space unmixing, per-channel phasor, and particle detection with per-particle lifetime and trajectory linking. It reuses the existing decay-phasor math (extended to per-pixel), CLSM reconstruction tooling, and the pixel-wise FLIM plumbing pattern, with all UIs delivered as AutoForm view specs.

# Status
Stub / unassigned (STATUS TABLE authoritative). Scope and reuse identified; implementation not started.

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 3).
- Shares phasor code with [PRD-55](prd-55.md) and may share spectral-unmixing code with [PRD-54](prd-54.md).
- Particle-tracking validation uses simulated stacks from [PRD-53](prd-53.md); UIs via [GUI & AutoForm](/subsystems/gui-autoform.md).

# Source
- Primary: `overhaul/PRD-52-phasor-flim-imaging.md`
