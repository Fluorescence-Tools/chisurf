# PRD-52 — Phasor-FLIM Imaging & Particle Tracking

Status: Stub · Parent: PRD-49 (§6 Phase 3) · Type: Feature
Related: PRD-38, PRD-40, PRD-49

## 1. Motivation

ChiSurf computes phasors of TCSPC *decays* but has no phasor-FLIM *imaging*: no
per-pixel phasor plot, universal semicircle navigation, ROI/threshold segmentation,
per-PIE/spectral-channel phasor, or phasor-based particle detection & tracking. PAM's
`Phasor`, `PhasorTIFF`, `ParticleDetection`, and `ParticleViewer` cover this.

## 2. Scope

- **Per-pixel phasor image** from CLSM/TTTR data; g,s maps with reference calibration.
- **Universal-circle interactive plot** (τ markers) with circular/rectangular/threshold
  ROI back-projection to the image; multi-population linear unmixing in phasor space.
- **Per-PIE-channel and spectral phasor.**
- **Particle detection & tracking** — segment particles from phasor/FLIM, per-particle
  lifetime (τ_phase, τ_mod), Hungarian/nearest-neighbour trajectory linking.

## 3. Reuse

- `chisurf/core/fluorescence/tcspc/phasor.py` — decay-phasor math (extend to per-pixel).
- `clsm` / tttrlib CLSM — image reconstruction, ROI/mask tooling.
- `img_pixel_mle` — pixel-wise FLIM plumbing pattern.
- skimage / scipy for segmentation + linking (or existing tracker if present).
- AutoForm + `view.json`; extend `image` section for the linked phasor↔image selection
  (add a reusable section if none fits — PRD-49 §2).

## 4. Acceptance

- Headless: per-pixel phasor of a synthetic single-lifetime FLIM stack lands on the
  universal circle at the expected coordinate; ROI in phasor space selects the correct
  pixels; two-component unmixing recovers known fractions.
- Particle tracking recovers known trajectories on a simulated moving-particle stack
  (PRD-53).

## 5. Non-goals

Spectral unmixing as a standalone spectroscopy tool (PRD-54, may share phasor code).
