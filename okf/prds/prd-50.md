---
type: PRD
prd: "50"
title: "PRD-50: Photon Distribution Analysis (PDA) Family"
description: Wrap the existing PDA histogram engine in ChiSurf models and AutoForm view specs, covering static distance-distribution PDA, dynamic/N-state kinetic PDA, error surfaces, three-color PDA, and a kinetic consistency check.
status: draft
phase: "unassigned"
resource: overhaul/PRD-50-pda-family.md
tags: [prd, fret]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
Photon Distribution Analysis fits the shot-noise-broadened FRET-efficiency histogram of single-molecule bursts to recover inter-dye distance distributions and, in its dynamic form, kinetic exchange between conformational states. The `tttrlib.Pda` C++ engine already computes the histograms, but no ChiSurf model, `view.json`, or plugin wraps it. This PRD adds the model + schema + AutoForm UI + fit integration in staged scope: static (single/multi-Gaussian, Lorentzian) PDA, dynamic/N-state kinetic PDA, Support-Plane/MCMC error surfaces, three-color PDA, and a kinetic consistency check. Dual-color models are already ported to the PRD-38 model/view-spec split with headless coverage; time-binned dynamic PDA, a 3-state variant, and PDA-specific error surfaces remain.

# Status
Draft / unassigned (STATUS TABLE authoritative). Dual-color static plus a dynamic two-state model done under AutoForm; several follow-ups (GUI light-path hook, time-binned dynamic PDA, 2D residual plot, PDA error surfaces) open.

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 1, first target).
- Wraps the `tttrlib.Pda` engine; consumes burst tables from the burst-selection pipeline.
- Renders via [GUI & AutoForm](/subsystems/gui-autoform.md); reads datasets through [Core target](/specs/core.md) rather than globals.

# Source
- Primary: `overhaul/PRD-50-pda-family.md`
