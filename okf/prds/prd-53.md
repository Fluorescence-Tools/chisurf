---
type: PRD
prd: "53"
title: "PRD-53: Simulation Workflow (Diffusion + FRET + Photon + Camera)"
description: Surface the existing Monte-Carlo simulation engines as a headless + AutoForm workflow producing synthetic ground truth for downstream analysis validation.
status: stub
phase: "unassigned"
resource: chisurf/core/fluorescence/simulation/
tags: [prd, simulation]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
ChiSurf owns the compute engines for Monte-Carlo fluorescence simulation — burst/diffusion simulator, a C++ decay/photon simulator, and a dye-diffusion quenching simulator — but exposes no surfaced workflow. This PRD wraps them as a headless + AutoForm simulator covering Brownian diffusion (2D/3D), distance-dependent FRET (static and dynamic/kinetic), photon emission with IRF/TCSPC, triplet/blinking/bleaching, and detector response, emitting TTTR streams, burst tables, decay histograms, and image/camera stacks. Beyond parity, it provides the reproducible, seed-deterministic synthetic ground truth that the PDA, imaging-correlation, and phasor-imaging acceptance tests depend on, with MFDB-registered synthetic datasets so downstream analyses stay provenance-tracked.

# Status
Stub / unassigned (STATUS TABLE authoritative). Engines exist (partial/engine-only); no surfaced workflow yet.

Parent: [PRD-49](prd-49.md) (Phase 4). Related: PRD-38, PRD-40, PRD-49; memory:
headless-mode-for-features.

# Motivation

The incumbent suite's simulation app generates Monte-Carlo fluorescence data
(Brownian diffusion, FRET, photophysics, TCSPC histograms, camera images) — invaluable
for validating analysis and teaching. ChiSurf has the engines (burbulator;
`chisurf/core/fluorescence/simulation/simulation_.cpp`; dye-diffusion quenching
simulator) but **no surfaced workflow**. Beyond parity, this simulator provides the
**synthetic ground truth** that PRD-50/51/52 acceptance tests depend on.

# Scope

- Headless + AutoForm simulation of: Brownian diffusion (2D/3D, box/concentration),
  distance-dependent FRET (static & dynamic/kinetic), photon emission with IRF/TCSPC,
  triplet/blinking/bleaching, detector response; outputs → TTTR stream, burst tables,
  decay histograms, and image/camera stacks.
- Parameter presets and reproducible seeds; MFDB-registered synthetic datasets so
  downstream analyses are provenance-tracked.

# Reuse

- burbulator + `chisurf/core/fluorescence/simulation/simulation_.cpp` (C++ engine).
- Existing acq/simulation device shim (`chisurf/plugins/core/acq/tcspc_devices/simulation/`).
- FRET/decay models (`chisurf/core/models/tcspc/*`) for ground-truth parameters.
- `ChiSurfAPI` headless entry; AutoForm + `view.json` UI (PRD-49 AutoForm mandate).

# Acceptance

- Headless: simulate a two-population smFRET dataset at set distances → run
  [PRD-50](prd-50.md) PDA → recover the input distances; simulate diffusion at known D →
  RICS/N&B ([PRD-51](prd-51.md)) recover D and brightness; simulate FLIM → phasor
  ([PRD-52](prd-52.md)) lands correctly.
- Deterministic given a fixed seed.

# Non-goals

Full MD simulation (external MD stays external; `traj_*` handles MD-derived FRET).

# Relationships
- Child of [PRD-49](prd-49.md) (Phase 4).
- Provides synthetic ground truth for [PRD-50](prd-50.md), [PRD-51](prd-51.md), and [PRD-52](prd-52.md).
- Headless entry via [Core target](/specs/core.md); UI via [GUI & AutoForm](/subsystems/gui-autoform.md).
