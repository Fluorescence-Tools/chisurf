---
type: PRD
prd: "54"
title: "PRD-54: Spectral Unmixing, pCF, nsFCS & FCCS Models"
description: Close the remaining spectroscopy gaps — spectral unmixing/spectral phasor, pair-correlation analysis, nanosecond-FCS/antibunching, a dedicated FCCS fit model, and a 2-photon FCS model.
status: stub
phase: "unassigned"
resource: overhaul/PRD-54-spectral-and-pcf.md
tags: [prd, spectroscopy, fcs]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD groups the remaining, individually small spectroscopy gaps versus the incumbent suite into a single long-tail phase: spectral unmixing / spectral-phasor decomposition of multi-channel spectral images, pair-correlation-function (pCF) analysis with carpet plots, a nanosecond-FCS / antibunching model, a dedicated dual-color FCCS fitting model (ChiSurf reads cross-correlations but has no FCCS fit model), and a 2-photon FCS model. Each reuses existing FCS model + `view.json` templates, `tttrlib` fine/ns-resolution correlators, and reference spectra, with all UIs delivered as AutoForm view specs. Acceptance is headless throughout — recovering known fractions, transport times, antibunching timescales, bound fractions, and diffusion coefficients on synthetic data.

# Status
Stub / unassigned (STATUS TABLE authoritative). Scope and reuse identified; implementation not started.

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 5).
- May share spectral-phasor code with [PRD-52](prd-52.md); pCF/FCS validation uses [PRD-53](prd-53.md) simulations.
- All UIs via [GUI & AutoForm](/subsystems/gui-autoform.md); image-correlation methods are [PRD-51](prd-51.md).

# Source
- Primary: `overhaul/PRD-54-spectral-and-pcf.md`
