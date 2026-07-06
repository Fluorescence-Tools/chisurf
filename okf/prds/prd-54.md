---
type: PRD
prd: "54"
title: "PRD-54: Spectral Unmixing, pCF, nsFCS & FCCS Models"
description: Close the remaining spectroscopy gaps — spectral unmixing/spectral phasor, pair-correlation analysis, nanosecond-FCS/antibunching, a dedicated FCCS fit model, and a 2-photon FCS model.
status: stub
phase: "unassigned"
resource: chisurf/core/models/fcs/
tags: [prd, spectroscopy, fcs]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
This PRD groups the remaining, individually small spectroscopy gaps versus the incumbent suite into a single long-tail phase: spectral unmixing / spectral-phasor decomposition of multi-channel spectral images, pair-correlation-function (pCF) analysis with carpet plots, a nanosecond-FCS / antibunching model, a dedicated dual-color FCCS fitting model (ChiSurf reads cross-correlations but has no FCCS fit model), and a 2-photon FCS model. Each reuses existing FCS model + `view.json` templates, `tttrlib` fine/ns-resolution correlators, and reference spectra, with all UIs delivered as AutoForm view specs. Acceptance is headless throughout — recovering known fractions, transport times, antibunching timescales, bound fractions, and diffusion coefficients on synthetic data.

# Status
Stub / unassigned (STATUS TABLE authoritative). Scope and reuse identified; implementation not started.

Parent: [PRD-49](prd-49.md) (Phase 5). Related: PRD-38, PRD-40, PRD-49.

# Motivation

Remaining spectroscopy gaps vs. the incumbent suite's spectral app, pair-correlation
app, and FCS model set: spectral unmixing / spectral phasor, pair-correlation-function
(pCF) analysis, nanosecond-FCS / antibunching model, and a dedicated dual-color FCCS
*fitting* model (ChiSurf reads cross-correlations but has no FCCS fit model).
Individually small; grouped here as the "long tail" phase.

# Scope

- **Spectral unmixing** — linear unmixing / spectral-phasor decomposition of multi-channel
  spectral images into species maps (may share [PRD-52](prd-52.md) phasor code).
- **pCF** — pair-correlation vs. distance from photon streams / coordinates; carpet plots.
- **nsFCS / antibunching** — ns-timescale correlation model (photon antibunching dip,
  fast conformational/photophysical dynamics).
- **FCCS fit model** — dual-color cross-correlation fitting (binding/co-diffusion).
- **2-photon FCS model** — the incumbent's 2-photon FCS model (γ=0.26, independent
  w_r/w_z); trivial add to `chisurf/core/models/fcs/`.

# Reuse

- `chisurf/core/models/fcs/` — model + view.json templates (nsFCS, FCCS, 2-photon).
- tttrlib correlators (fine/ns time resolution) for nsFCS and pCF.
- `spectra_downloader` / spectral plugins — reference spectra for unmixing.
- [PRD-52](prd-52.md) phasor code — spectral phasor.
- AutoForm + `view.json` for all UIs (PRD-49 AutoForm mandate).

# Acceptance

- Headless: unmix two known spectra → correct fractions; pCF on simulated diffusion
  ([PRD-53](prd-53.md)) recovers transport time vs. distance; nsFCS recovers a known
  antibunching timescale; FCCS model recovers a known bound fraction; 2-photon FCS
  recovers D on synthetic 2p data.

# Non-goals

Phasor-FLIM *imaging* ([PRD-52](prd-52.md)); image-correlation methods
([PRD-51](prd-51.md)).

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 5).
- May share spectral-phasor code with [PRD-52](prd-52.md); pCF/FCS validation uses [PRD-53](prd-53.md) simulations.
- All UIs via [GUI & AutoForm](/subsystems/gui-autoform.md); image-correlation methods are [PRD-51](prd-51.md).
