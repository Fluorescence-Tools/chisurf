# PRD-54 — Spectral Unmixing, pCF, nsFCS & FCCS Models

Status: Stub · Parent: PRD-49 (§6 Phase 5) · Type: Feature
Related: PRD-38, PRD-40, PRD-49

## 1. Motivation

Remaining spectroscopy gaps vs. PAM's `Spectral`, `PCFAnalysis`, and FCS model set:
spectral unmixing / spectral phasor, pair-correlation-function (pCF) analysis,
nanosecond-FCS / antibunching model, and a dedicated dual-color FCCS *fitting* model
(ChiSurf reads cross-correlations but has no FCCS fit model). Individually small; grouped
here as the "long tail" phase.

## 2. Scope

- **Spectral unmixing** — linear unmixing / spectral-phasor decomposition of multi-channel
  spectral images into species maps (may share PRD-52 phasor code).
- **pCF** — pair-correlation vs. distance from photon streams / coordinates; carpet plots.
- **nsFCS / antibunching** — ns-timescale correlation model (photon antibunching dip,
  fast conformational/photophysical dynamics).
- **FCCS fit model** — dual-color cross-correlation fitting (binding/co-diffusion).
- **2-photon FCS model** — PAM `develop`'s `FCS_D_2Photon` (γ=0.26, independent w_r/w_z);
  trivial add to `chisurf/core/models/fcs/`.

## 3. Reuse

- `chisurf/core/models/fcs/` — model + view.json templates (nsFCS, FCCS, 2-photon).
- tttrlib correlators (fine/ns time resolution) for nsFCS and pCF.
- `spectra_downloader` / spectral plugins — reference spectra for unmixing.
- PRD-52 phasor code — spectral phasor.
- AutoForm + `view.json` for all UIs (PRD-49 §2).

## 4. Acceptance

- Headless: unmix two known spectra → correct fractions; pCF on simulated diffusion (PRD-53)
  recovers transport time vs. distance; nsFCS recovers a known antibunching timescale; FCCS
  model recovers a known bound fraction; 2-photon FCS recovers D on synthetic 2p data.

## 5. Non-goals

Phasor-FLIM *imaging* (PRD-52); image-correlation methods (PRD-51).
