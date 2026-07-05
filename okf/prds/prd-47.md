---
type: PRD
prd: "47"
title: "PRD-47: Relocate Spectroscopy Physics into an External Biophysical Modeling Framework"
description: Consolidate duplicated fluorescence-spectroscopy physics so an external biophysical modeling framework becomes the single home, reducing ChiSurf to fitting-model glue, GUI, and data-IO that calls into it.
status: draft
phase: "cross-cutting"
resource: overhaul/PRD-47-spectroscopy-to-impbff.md
tags: [prd, core]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The same fluorescence-spectroscopy physics (decay convolution, FRET conversions, distributions, Förster-radius math, anisotropy, phasor, FCS curve algebra) is implemented in several repositories at once, so code paths drift and fixes land in only one place. This PRD makes an external biophysical modeling framework (via its Python staging layer) the single home for spectroscopy physics: delete ChiSurf copies that already exist upstream, relocate ChiSurf-only physics that belongs upstream, and keep fitting-model parameter groups, plugin GUIs, and data-IO in ChiSurf but delegate numerics upstream. Photon-stream processing (burst search, PDA, raw correlation) stays with `tttrlib`, deliberately outside the modeling framework. A thin import seam (`chisurf/core/fluorescence/_backend.py`) plus a numeric-equivalence test gates every swap so there is no behavioural regression.

# Status
Draft / cross-cutting (STATUS TABLE authoritative). No code changes yet; defines inventory, boundary, and phased migration order.

# Relationships
- Independent refactor targeting the [Core target](/specs/core.md) fluorescence layer.
- Draws an explicit boundary: burst/PDA/raw-correlation stay with `tttrlib`, relevant to [PRD-50](prd-50.md) and [PRD-53](prd-53.md).
- Touches the fitting-model/AutoForm seam described in [GUI & AutoForm](/subsystems/gui-autoform.md) without changing it.

# Source
- Primary: `overhaul/PRD-47-spectroscopy-to-impbff.md`
