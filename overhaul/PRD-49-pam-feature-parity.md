# PRD-49 — PAM Feature Parity (and Beyond)

Status: Draft · Owner: ChiSurf core · Type: Roadmap / master PRD
Related: PRD-38 (model/view-spec split), PRD-40 (declarative dataset editors),
PRD-28 (ndxplorer burst), PRD-04 (burst pipeline), PRD-26 (model-driven data layer)

## 1. Context & goal

**PAM** ("PIE Analysis with MATLAB", Schrimpf/Barth/Hendrix/Lamb, *Biophys. J.*
2018; GitLab `PAM-PIE/PAM`) is the incumbent integrated suite for quantitative
fluorescence microscopy and spectroscopy — FCS/FCCS/pCF, smFRET/MFD, phasor-FLIM,
and image-correlation (ICS/RICS/tICS/N&B). It is a ~730-file MATLAB codebase across
~13 GUI apps. A local checkout lives at `thirdparty/PAM` (branches `master` for
MATLAB R2022b–R2024b, `develop` for R2025a+).

**Goal.** Reach *feature parity* with PAM across every analysis modality, then
*surpass* it — leveraging ChiSurf's advantages (open Python/Qt, plugin system,
headless/CLI/server modes, MFDB provenance, deep structural-FRET integration,
standardized data exchange). ChiSurf already matches or beats PAM in most areas;
this PRD gives an honest parity matrix, enumerates the real gaps, and lays out a
phased roadmap that spawns per-module sub-PRDs (PRD-50…54).

This is a **roadmap PRD**: it decides *what* and *in what order*, and delegates the
*how* to sub-PRDs. It does not itself change code.

## 2. Cross-cutting mandate — AutoForm + JSON for all new work

**Every new model and analysis UI in this roadmap MUST be built as a declarative
model + `view.json` rendered by AutoForm — never hand-written Qt widgets.** This is
a hard requirement inherited by all sub-PRDs.

- A new analysis ships as: a model class under `chisurf/core/models/<family>/`
  (+ `chisurf/core/dataspec/` schema) paired with a `view.json` consumed by
  `chisurf/gui/autoform/`.
- Reuse existing AutoForm sections (`parameter_table`, `image`, `waterfall`,
  `path_list`, `wizard`, `info`, `embed`, …). If a needed section is missing, **add
  it to `chisurf/gui/autoform/sections/`** so every plugin benefits — do not build a
  one-off widget.
- Follows **PRD-38** (model/view-spec split) and **PRD-40** (declarative editors);
  verified headlessly via the `test-model-editor` skill.
- We do **not** port PAM's MATLAB GUIs to Qt — we re-express each capability as a
  data-driven view spec over a reusable compute core (tttrlib / chinet / chisurf.core).

## 3. Method-by-method parity matrix

Status legend: **PARITY+** (ChiSurf ahead) · **PARITY** · **PARTIAL** (exists but
incomplete) · **ENGINE-ONLY** (compute core present, no ChiSurf model/UI) ·
**ABSENT**.

### PAM hub (data browser, correlation, microtime, image, PIE config)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| TTTR read-in (ptu/ht3/pt3/spc/PhotonHDF5/T3R) | PARITY+ | tttrlib readers; `tttr_toolbox`, staging cache |
| PIE / µsALEX channel & detector setup | PARITY | MFDB `mfdb_setup_pie_window`, `burst_selection`, `ptu_alex_creator` |
| Correlation (auto/cross) from stream | PARITY | tttrlib correlators; `burst_fcs_correlator`, `fcs_*` |
| CLSM image from stream (intensity, mean µtime) | PARITY | `clsm` (CLSM-Draw), `tttr_image_browser` |
| Unified "one-window" data browser hub | PARTIAL | Capabilities exist but spread across plugins; no single PAM-style hub (acceptable — different UX philosophy) |
| Native Zeiss CZI / Leica LIF import | ABSENT | Zeiss/Leica only via PTU today → PRD-49 backlog |

### BurstBrowser (smFRET / MFD)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Burst search & selection | PARITY+ | `burst_selection`: BOCPD/Kalman/CUSUM (PAM is threshold-only) |
| E / S / proximity-ratio, corrections (γ, crosstalk, dir-exc, bg) | PARITY | `burst_selection`, `burst_bva` |
| MFD 2D histograms (E–τ, S–τ, E–S) + FRET lines | PARITY | `ndxplorer` + `fret_line` |
| Burst Variance Analysis (BVA) | PARITY | `burst_bva`, `chisurf/core/fluorescence/burst/bva.py` |
| Burst-wise MLE lifetimes | PARITY | `burst_mle_analysis` |
| Burst-wise phasor lifetimes | PARTIAL | decay-level phasor exists; not wired burst-wise → PRD-52 |
| 3-color (3c-MFD) FRET | ABSENT | 2-color only today → PRD-49 backlog / PRD-50 (3c) |
| FRET-2CDE / ALEX-2CDE dynamics filters | ABSENT | → PRD-49 backlog |
| Kinetic consistency check (dynamic-state resampling) | ABSENT | → PRD-50 (shares kinetic-PDA machinery) |

### FCSFit (FCS/FCCS fitting)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| FCS diffusion models (2D/3D, anomalous, triplet, flow, scanning, 2-focus, afterpulsing, bleaching) | PARITY | 15+ models in `chisurf/core/models/fcs/` |
| 2-photon FCS model (new on PAM `develop`) | ABSENT | small add → PRD-49 backlog |
| FCCS dual-color *fitting* model | PARTIAL | cross-corr read/compute yes; dedicated FCCS fit model no → PRD-54 |
| fFCS (lifetime-filtered FCS) | PARITY | `fcs_filter_calculator`, `flc_2d` |
| nsFCS / antibunching model | ABSENT | → PRD-54 |
| Session save/load, multi-file global | PARITY | `globalview`, project/MFDB state |

### TauFit (lifetime / TCSPC)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Multi-exp / stretched-exp decay fits | PARITY | `chisurf/core/models/tcspc/*`, `lifetime_analysis` |
| Time-resolved anisotropy (1/2/4 corr. times) | PARITY | `tcspc/anisotropy.py`, `tr_anisotropy` plugin |
| Donor/acceptor global fit w/ distance distributions | PARITY | `fret.py`, `maxent_decay` |
| IRF conv. (Gauss/Gamma), MLE, MCMC error | PARITY+ | `irf_estimator`, `fitting/sample.py`; plus MaxEnt/LLTF PAM lacks |
| Sub-ensemble TCSPC | PARITY | via burst microtime histograms |

### PDAFit (Photon Distribution Analysis)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Static Gaussian/Lorentzian distance-distribution PDA | ENGINE-ONLY | `tttrlib.Pda` engine exists; **no ChiSurf model/UI** → PRD-50 |
| Dynamic PDA / 2-3-state kinetic networks | ABSENT | → PRD-50 |
| Error surfaces (Support-Plane, MCMC, Hessian) | PARTIAL | generic `fitting/sample.py`; not wired to PDA → PRD-50 |

### tcPDA (three-color PDA)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| 1D/2D/3D three-color distance-distribution PDA, time-binned, Bayesian | ABSENT | → PRD-50 (later stage) |

### Phasor / PhasorTIFF (phasor-FLIM)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Phasor of TCSPC decays (g, s) | PARITY | `chisurf/core/fluorescence/tcspc/phasor.py` |
| Per-pixel phasor image + universal circle + ROI segmentation | ABSENT | → PRD-52 |
| Per-PIE-channel phasor | ABSENT | → PRD-52 |
| Spectral phasor | ABSENT | → PRD-52 / PRD-54 |

### Mia / MIAFit (image correlation)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| RICS (+ triplet/blinking, immobile, flow) | PARITY | `chisurf/core/models/rics/`, `experiments/rics/` |
| ICS (2D spatial) | PARTIAL | subsumed by RICS core |
| tICS / STICS (temporal / spatiotemporal) | ABSENT | → PRD-51 |
| Number & Brightness (N&B) | ABSENT | → PRD-51 |
| iMSD (free/mob-immob/blinking) | ABSENT | → PRD-51 |
| Crosstalk-free spectral RICS | ABSENT | → PRD-51 |

### Spectral
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Spectral image load, spectral phasor, unmixing | ABSENT | → PRD-54 |

### ParticleDetection / ParticleViewer
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Phasor-FLIM particle detection, segmentation, tracking | ABSENT | → PRD-52 |

### PCFAnalysis
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Pair-correlation function (pCF) analysis | ABSENT | → PRD-54 |

### Sim (Monte-Carlo simulation)
| PAM capability | Status | ChiSurf location / note |
|---|---|---|
| Diffusion + FRET + photon + camera simulation | PARTIAL / ENGINE-ONLY | burbulator + `chisurf/core/fluorescence/simulation/simulation_.cpp`; no surfaced workflow → PRD-53 |

## 4. Where ChiSurf already beats PAM

These are net differentiators to *keep and lean into*, not gaps:

- **Structural-FRET integration PAM has essentially none of:** accessible-volume
  modelling (`fps_json_editor`, IMP.bff → MRC densities), FRET-restrained rigid-body
  **docking & library screening** (`fret_docking`, IMP.pmi), κ² distributions
  (`kappa2_dist`), static/dynamic/WLC FRET lines (`fret_line`), molecular viewer
  (`chimol`), HYDROPRO diffusion (`hydropro`), MD-trajectory FRET (`traj_*`).
- **Provenance & data management:** MFDB lineage/audit/ACL (`chisurf/core/mfdb/`),
  PDB-IHM / FLR-CIF standardized exchange, sample/reagent database — PAM stores loose
  session files only.
- **Modern burst detection:** Bayesian (BOCPD), Kalman, CUSUM change-point search
  vs. PAM's fixed thresholds.
- **Model-free lifetime:** MaxEnt (`maxent_decay`) and LLTF (`lltf`).
- **Deployment:** headless/CLI/server modes via `ChiSurfAPI` (local/hybrid/server),
  ZMQ/JSON-RPC server, plugin architecture, open-source Python — vs. MATLAB license +
  GUI-only.

Design principle for parity work: **reach parity through the data-driven
model+view.json framework, so each new module is also headless-testable and
provenance-tracked — i.e. we surpass PAM on the way to matching it.**

## 5. Gap backlog

Ordered within each phase by reuse leverage. "Reuse" names the existing asset the
implementer builds on.

| Gap | What PAM does | Reuse in ChiSurf | Effort | Owner |
|---|---|---|---|---|
| Static distance-distribution PDA | E-histogram shot-noise model, Gaussian/Lorentzian | `tttrlib.Pda` engine, `models/fcs` template | M | PRD-50 |
| Dynamic / N-state kinetic PDA | 2-3 state kinetic-network PDA | above + kinetic scheme spec | L | PRD-50 |
| PDA error surfaces (SPA/MCMC) | support-plane, MCMC, Hessian | `fitting/sample.py` | M | PRD-50 |
| Three-color tcPDA | 1/2/3D 3c distance distributions | PDA core once built | L | PRD-50 |
| Kinetic consistency check | dynamic-state burst resampling | burst tables + kinetic-PDA sim | M | PRD-50 |
| N&B | brightness/aggregation from image fluctuations | `clsm`, RICS core | M | PRD-51 |
| tICS / STICS | temporal / spatiotemporal ICS | RICS/ICS core | M | PRD-51 |
| iMSD | MSD from image correlation | RICS models | M | PRD-51 |
| Spectral RICS | crosstalk-free spectral weighting | RICS core + spectral channels | M | PRD-51 |
| Per-pixel phasor-FLIM imaging | phasor image, universal circle, ROI seg | `tcspc/phasor.py`, `clsm` | M | PRD-52 |
| Spectral/PIE-channel phasor | phasor per spectral/PIE channel | phasor core | M | PRD-52 |
| Phasor particle detection/tracking | segment + Hungarian tracker | phasor imaging + skimage | L | PRD-52 |
| Simulation workflow | MC diffusion+FRET+photon+camera | burbulator, `simulation_.cpp` | M | PRD-53 |
| Spectral unmixing | spectral phasor / linear unmixing | spectra plugins, phasor | M | PRD-54 |
| pCF | pair-correlation from stream/coords | tttrlib correlators | M | PRD-54 |
| nsFCS / antibunching model | ns-timescale correlation model | `models/fcs`, tttrlib fine correlator | S | PRD-54 |
| FCCS dual-color fit model | cross-corr fitting | `models/fcs` | S | PRD-54 |
| 2-photon FCS model | γ=0.26, indep. w_r/w_z | `models/fcs` | S | PRD-49 backlog |
| 3c-MFD / FRET-2CDE / ALEX-2CDE | dynamics filters, 3c histograms | `ndxplorer`, burst core | M | PRD-49 backlog |
| Native CZI / LIF import | Zeiss/Leica file readers | staging/fio, aicsimageio/readlif candidates | M | PRD-49 backlog |

## 6. Phased roadmap

1. **Phase 1 — PDA family** (PRD-50) *(first target)*. Highest reuse: `tttrlib.Pda`
   already computes the histograms. Static PDA model → dynamic/N-state kinetic PDA →
   SPA/MCMC errors → three-color tcPDA → kinetic-consistency-check. Core smFRET
   differentiator; unblocks burst-dynamics analysis.
2. **Phase 2 — Imaging correlation** (PRD-51). Biggest single ABSENT cluster; extends
   the existing RICS/clsm core: N&B, tICS/STICS, iMSD, spectral RICS.
3. **Phase 3 — Phasor-FLIM imaging** (PRD-52). Per-pixel phasor, universal circle, ROI
   segmentation, spectral/PIE-channel phasor, particle detection/tracking.
4. **Phase 4 — Simulation workflow** (PRD-53). Surfaces the burbulator/C++ engine as a
   headless+AutoForm simulator — provides synthetic ground truth to validate Phases 1-3.
5. **Phase 5 — Spectral, pCF, nsFCS/FCCS** (PRD-54). Remaining spectroscopy gaps.
6. **Rolling backlog** (this PRD): 2-photon FCS model, 3c-MFD / 2CDE filters, CZI/LIF
   import — small/independent, land opportunistically.

## 7. Designated sub-PRDs

| PRD | Scope | First acceptance signal |
|---|---|---|
| PRD-50 | PDA family (static → kinetic → tcPDA → consistency check) | headless fit of synthetic 2-Gaussian E-histogram recovers input distances |
| PRD-51 | Imaging correlation: N&B, tICS/STICS, iMSD, spectral RICS | N&B recovers brightness on simulated stack |
| PRD-52 | Phasor-FLIM imaging + particle tracking | per-pixel phasor of synthetic FLIM lands on universal circle |
| PRD-53 | Simulation workflow (diffusion+FRET+photon+camera) | headless sim → burst table reproduces set E |
| PRD-54 | Spectral unmixing, pCF, nsFCS, FCCS fit model | unmix two known spectra; pCF on simulated diffusion |

## 8. Verification strategy

- Every new model is exercised by a **headless test** (no GUI), following the
  `test-model-editor` skill and the `test/`, `**/test/` patterns. See CLAUDE.md
  "headless test path for every feature".
- Compute cores are reused, not reimplemented: `tttrlib` (Pda, correlators, CLSM),
  `chinet`, `chisurf/core/fluorescence/*`. Sub-PRDs must state which engine call they
  wrap.
- Ground truth from Phase-4 simulation validates Phases 1-3 quantitatively.
- Each sub-PRD's *acceptance* section names a concrete synthetic-data assertion.

## 9. Non-goals

- Not reimplementing PAM's MATLAB UI or shipping a compiled MATLAB-free clone.
- Not a single monolithic "PAM hub" window — ChiSurf stays plugin/AutoForm-driven.
- No new bespoke Qt widgets for analysis UIs (see §2 mandate).
- This PRD implements nothing; it authorizes PRD-50…54 and the rolling backlog.
