---
type: PRD
prd: "53"
title: "PRD-53: Simulation Workflow (Diffusion + FRET + Photon + Camera)"
description: Surface the existing Monte-Carlo simulation engines as a headless + AutoForm workflow producing synthetic ground truth for downstream analysis validation.
status: stub
phase: "unassigned"
resource: overhaul/PRD-53-simulation-workflow.md
tags: [prd, simulation]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
ChiSurf owns the compute engines for Monte-Carlo fluorescence simulation — burst/diffusion simulator, a C++ decay/photon simulator, and a dye-diffusion quenching simulator — but exposes no surfaced workflow. This PRD wraps them as a headless + AutoForm simulator covering Brownian diffusion (2D/3D), distance-dependent FRET (static and dynamic/kinetic), photon emission with IRF/TCSPC, triplet/blinking/bleaching, and detector response, emitting TTTR streams, burst tables, decay histograms, and image/camera stacks. Beyond parity, it provides the reproducible, seed-deterministic synthetic ground truth that the PDA, imaging-correlation, and phasor-imaging acceptance tests depend on, with MFDB-registered synthetic datasets so downstream analyses stay provenance-tracked.

# Status
Stub / unassigned (STATUS TABLE authoritative). Engines exist (partial/engine-only); no surfaced workflow yet.

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 4).
- Provides synthetic ground truth for [PRD-50](prd-50.md), [PRD-51](prd-51.md), and [PRD-52](prd-52.md).
- Headless entry via [Core target](/specs/core.md); UI via [GUI & AutoForm](/subsystems/gui-autoform.md).

# Source
- Primary: `overhaul/PRD-53-simulation-workflow.md`
