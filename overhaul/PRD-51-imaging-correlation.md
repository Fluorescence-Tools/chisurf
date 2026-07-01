# PRD-51 — Imaging Correlation: N&B, tICS/STICS, iMSD, Spectral RICS

Status: Stub · Parent: PRD-49 (§6 Phase 2) · Type: Feature
Related: PRD-38, PRD-40, PRD-49

## 1. Motivation

The largest single ABSENT cluster vs. PAM's `Mia`/`MIAFit`. ChiSurf has RICS (≈2D-ICS)
but lacks Number & Brightness, temporal/spatiotemporal ICS, iMSD, and crosstalk-free
spectral RICS. These extract oligomerization state, transport maps, and mobility from
image stacks — core quantitative-imaging methods.

## 2. Scope

- **N&B** — apparent/true brightness & number from pixel intensity mean/variance;
  aggregation-state maps.
- **tICS / STICS** — temporal and spatiotemporal image correlation (flow/transport).
- **iMSD** — mean-square-displacement from image correlation: free / mobile-immobile /
  blinking variants (PAM `iMSD_*` models).
- **Spectral RICS** — spectral-weighting for crosstalk-free multi-color RICS.

## 3. Reuse

- `chisurf/core/models/rics/`, `chisurf/core/experiments/rics/ics_core.py` — correlation core.
- `clsm` / tttrlib CLSM — image-from-stream + ROI/mask handling.
- PAM `Models/miafit/*.miafit` — reference model forms (2D Gaussian, iMSD, RICS variants).
- AutoForm + `view.json` for all UIs (PRD-49 §2); reuse `image`/`waterfall`/ROI sections.

## 4. Acceptance

- Headless: N&B recovers known brightness on a simulated fluctuating stack; iMSD recovers
  a known D; STICS recovers a known flow vector. Synthetic stacks from PRD-53 simulator.
- Models render from `view.json` with no bespoke Qt; selectable in add-fit; model-editor
  headless checks pass.

## 5. Non-goals

Phasor imaging (PRD-52); spectral unmixing beyond RICS weighting (PRD-54).
