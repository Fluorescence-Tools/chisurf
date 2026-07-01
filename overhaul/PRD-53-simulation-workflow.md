# PRD-53 — Simulation Workflow (Diffusion + FRET + Photon + Camera)

Status: Stub · Parent: PRD-49 (§6 Phase 4) · Type: Feature
Related: PRD-38, PRD-40, PRD-49; memory: headless-mode-for-features

## 1. Motivation

PAM's `Sim` generates Monte-Carlo fluorescence data (Brownian diffusion, FRET,
photophysics, TCSPC histograms, camera images) — invaluable for validating analysis
and teaching. ChiSurf has the engines (burbulator; `chisurf/core/fluorescence/
simulation/simulation_.cpp`; dye-diffusion quenching simulator) but **no surfaced
workflow**. Beyond parity, this simulator provides the **synthetic ground truth** that
PRD-50/51/52 acceptance tests depend on.

## 2. Scope

- Headless + AutoForm simulation of: Brownian diffusion (2D/3D, box/concentration),
  distance-dependent FRET (static & dynamic/kinetic), photon emission with IRF/TCSPC,
  triplet/blinking/bleaching, detector response; outputs → TTTR stream, burst tables,
  decay histograms, and image/camera stacks.
- Parameter presets and reproducible seeds; MFDB-registered synthetic datasets so
  downstream analyses are provenance-tracked.

## 3. Reuse

- burbulator + `chisurf/core/fluorescence/simulation/simulation_.cpp` (C++ engine).
- Existing acq/simulation device shim (`chisurf/plugins/core/acq/tcspc_devices/simulation/`).
- FRET/decay models (`chisurf/core/models/tcspc/*`) for ground-truth parameters.
- `ChiSurfAPI` headless entry; AutoForm + `view.json` UI (PRD-49 §2).

## 4. Acceptance

- Headless: simulate a two-population smFRET dataset at set distances → run PRD-50 PDA →
  recover the input distances; simulate diffusion at known D → RICS/N&B (PRD-51) recover D
  and brightness; simulate FLIM → phasor (PRD-52) lands correctly.
- Deterministic given a fixed seed.

## 5. Non-goals

Full MD simulation (external MD stays external; `traj_*` handles MD-derived FRET).
