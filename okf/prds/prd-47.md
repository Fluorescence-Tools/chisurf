---
type: PRD
prd: "47"
title: "PRD-47: Relocate Spectroscopy Physics into an External Biophysical Modeling Framework"
description: Consolidate duplicated fluorescence-spectroscopy physics so an external biophysical modeling framework becomes the single home, reducing ChiSurf to fitting-model glue, GUI, and data-IO that calls into it.
status: draft
phase: "cross-cutting"
resource: chisurf/core/fluorescence/
tags: [prd, core]
timestamp: '2026-07-05T00:00:00Z'
---

# Summary
The same fluorescence-spectroscopy physics (decay convolution, FRET conversions, distributions, Förster-radius math, anisotropy, phasor, FCS curve algebra) is implemented in several repositories at once, so code paths drift and fixes land in only one place. This PRD makes an external biophysical modeling framework (via its Python staging layer) the single home for spectroscopy physics: delete ChiSurf copies that already exist upstream, relocate ChiSurf-only physics that belongs upstream, and keep fitting-model parameter groups, plugin GUIs, and data-IO in ChiSurf but delegate numerics upstream. Photon-stream processing (burst search, PDA, raw correlation) stays with `tttrlib`, deliberately outside the modeling framework. A thin import seam (`chisurf/core/fluorescence/_backend.py`) plus a numeric-equivalence test gates every swap so there is no behavioural regression.

# Status
Draft / cross-cutting (STATUS TABLE authoritative). No code changes yet; defines inventory, boundary, and phased migration order.

# Problem
The same fluorescence-spectroscopy physics is implemented in **several** places at once:

- The external biophysical modeling framework — a compiled C++/SWIG core (decay convolution, scoring, AV/FRET, photon statistics). The intended home.
- The framework's **Python staging layer** — a PEP-420 namespace-package mixin that adds Python enhancements on top of the compiled core *without* touching the upstream framework. The intended staging ground for new Python physics.
- `chisurf` — a desktop fitter that carries its **own** parallel copies of decay convolution, FRET conversions, distributions, Förster-radius math, etc.
- `quest` — a forward simulator that already consumes the framework's AV correctly.

ChiSurf reimplements physics that the framework / its staging layer already provide (true duplication), and also hosts physics that *should* live in the framework but does not yet (decay-side gaps). The result: two code paths drift, bug-fixes land in one place only, and `quest`/`chisurf`/the framework all disagree on, e.g., the Förster-radius prefactor.

**Goal of this PRD:** make the modeling framework (via its Python staging layer for the Python layer) the *single* home for spectroscopy physics; reduce ChiSurf to fitting-model glue, GUI, and data-IO that *calls into* the framework.

# Goals
1. **Delete every ChiSurf algorithm that already exists** in the framework or its staging layer; replace with an import (the 🔴 set below).
2. **Relocate ChiSurf-only physics** that belongs in the framework into its Python staging layer, then import it back (the 🟡 set).
3. **Keep fitting-model parameter groups, plugin GUIs, and data IO in ChiSurf** (the 🟢 set) — but have them delegate numerics to the framework.
4. **Draw an explicit boundary**: photon-stream processing (burst search, PDA, raw correlation) stays with `tttrlib`, *not* the framework (the ⚪ set).
5. **No behavioural regression** — ChiSurf's existing test suite stays green; a numeric-equivalence test guards each swap.

# Non-goals
- Rewriting the framework's C++ core. New Python physics lands in the staging layer first; promotion into compiled C++ is a later, separate decision.
- Moving burst detection / PDA / multi-tau correlation into the framework (they are `tttrlib`-aligned single-molecule stream processing).
- Touching `quest`'s forward simulator beyond noting its overlap with the framework's coarse-grained dye system (tracked as a follow-up, not in this PRD).
- Changing ChiSurf's fitting framework, AutoForm, or plugin manifests.

# Legend

| Mark | Meaning | Action |
|------|---------|--------|
| 🔴 DUP | Already in the framework / its staging layer | Delete ChiSurf copy, import shared |
| 🟡 MOVE | Real physics, gap in the framework | Relocate to the framework's staging layer, import back |
| 🟢 STAY | ChiSurf fitting/model/GUI glue | Keep; delegate numerics to the framework |
| ⚪ tttrlib | Photon-stream processing | Leave with `tttrlib`; out of scope for the framework |

# What the destination already provides
**Framework (compiled C++/SWIG):** `DecayCurve`, `DecayConvolution` (6 kernels: FAST/AVX/periodic/arbitrary-time-axis), `DecayLifetimeHandler`, `DecayScale`, `DecayPileup` (Coates), `DecayPattern` (scatter), `DecayLinearization`, `DecayScore` (7 χ² types), `DecayRoutines` (`decay_fconv*`, `decay_rescale*`, `shift_lamp`, pile-up, discriminate), `PhotonStatistics` (neyman/poisson/pearson/gauss/cnp/sswr + **polarized** wcm/twoIstar), `AV` / `AVNetworkRestraint` / `PathMap` / `FPSReaderWriter`, and `fret_efficiency` / `distance_fret` / `av_distance` / `av_distance_distribution`.

**Framework Python staging layer (namespace-merged):** `distance_metrics.py` (FRET E↔R, asymmetric χ², AV-pair stats), `polymer.py` (Gaussian / worm-like-chain linkers), `distributions.py` (normal, generalized-normal, poisson_0toN), `fps.py` (fps.json IO), `av/` (BasicAV, ACV, numba kernels), `label/`, `restraints/`, and the large `cgdye/` system (rotamer FRET, **R₀ from spectra**, static/dynamic/dynamic+ regimes, exact kinetic master-equation, κ²).

# Scope — ChiSurf spectroscopy inventory & disposition

## TCSPC / fluorescence decay

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/tcspc/convolve.py` (`convolve_lifetime_spectrum_nb`, periodic, `convolve_decay_nb`) | 🔴 DUP | framework `DecayConvolution` / `decay_fconv*` |
| `core/fluorescence/tcspc/tcspc.py` → `rescale_w_bg` | 🔴 DUP | framework `DecayScale` / `decay_rescale_w_bg` |
| `core/fluorescence/tcspc/tcspc.py` → `bin_lifetime_spectrum`, `pddem` | 🟡 MOVE | framework staging `decay` (spectrum helpers) |
| `core/fluorescence/general.py` (`fretrate_to_distance`, `distance↔fret_efficiency/rate`, `lifetime↔fret_efficiency`, `compute_mean_fret`, `combine_interleaved_spectra`) | 🔴 DUP | framework `fret_efficiency`/`distance_fret` + staging `distance_metrics.py` |
| `core/fluorescence/general.py` → `calculate_fluorescence_decay`, `fret_induced_donor_decay` | 🟡 MOVE | framework staging `decay` (builds on `DecayConvolution`) |
| `core/fluorescence/tcspc/irf.py`, `irf_estimation.py` (synthetic IRF, rising-edge, RL deconvolution) | 🟡 MOVE | new framework staging `irf` (its `generalized_normal_distribution` is 🔴 DUP of staging `distributions.py`) |
| `core/fluorescence/tcspc/corrections.py` (`compute_linearization_table`) | 🟡 MOVE | next to framework `DecayLinearization` (which applies, but can't compute, the table) |
| `core/fluorescence/tcspc/phasor.py` | 🟡 MOVE | new framework staging `phasor` |
| `core/math/functions/distributions.py` (`poisson_0toN`, `normal_distribution`, `generalized_normal_distribution`) | 🔴 DUP | **exact** duplicate of the framework staging `distributions.py` |
| `core/models/tcspc/*` (lifetime, fret, anisotropy, mix_model, nusiance, av_decay, maxent, pddem, fret_structure) | 🟢 STAY | fitting-model parameter groups; back with framework kernels (the MaxEnt solver is a MOVE candidate) |

## FRET

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/fret/forster.py` (`overlap_integral`, `forster_radius`, `forster_radius_from_spectra`) | 🔴 DUP | consolidate with framework staging `cgdye/rotamer/r0.py` (already computes R₀ from spectra) |
| `core/fluorescence/fret/__init__.py` (`apparent_fret_efficiency`, `fret_efficiency`, `fret_efficency_to_fdfa`) | 🔴 DUP | framework `fret_efficiency` + staging `distance_metrics.py` |
| `core/fluorescence/fret/acceptor.py` (`da_a0_to_ad`) | 🟡 MOVE | framework staging `fret` (uses convolution already in the framework) |
| `core/fluorescence/fret/fret_line.py` + `plugins/fret_line` | 🟢 STAY | model-sweep generator; algorithm core could MOVE, GUI stays |

## Anisotropy

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/anisotropy/kappa2.py` (`kappasq*`, `kappasq_dwt`, `p_isotropic_orientation_factor`, `s2delta`) | 🟡 MOVE | consolidate with `cgdye` κ² → framework staging `kappa2` |
| `core/fluorescence/anisotropy/decay.py` (`vm_rt_to_vv_vh`) | 🟡 MOVE | framework staging `anisotropy` |
| `core/fluorescence/anisotropy/integrals.py` (Perrin anisotropy, G-factor) | 🟡 MOVE | framework staging `anisotropy` |
| `plugins/jordi_anisotropy`, `jordi_g_factor`, `kappa2_dist` | 🟢 STAY | GUI wrappers over the MOVE targets above |

## FCS (largest physics gap in the framework today)

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/fcs/correlate.py` (multi-tau, numba) | ⚪ tttrlib | prefer `tttrlib`'s correlator over a second copy |
| `core/fluorescence/fcs/merge.py`, `normalization.py`, `filtered.py`, `channel_setups.py` | 🟡 MOVE | new framework staging `fcs` (curve algebra) |
| `core/models/fcs/maxent.py` (`build_diffusion_kernel`, L-curve MEM) | 🟡 MOVE | framework staging `fcs` |
| `plugins/fcs/flc_2d/fit/*` (ILT, MEM 1D/2D, kinetics) | 🟡 MOVE | framework staging `fcs`; rest of `plugins/fcs/*` is GUI 🟢 |

## Burst / PDA / PCH / RICS

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/burst/*` (filter, background, cusum, bocpd, kalman, bva) | ⚪ tttrlib | stays — stream processing |
| `plugins/burst/*` | 🟢 STAY | GUI + `tttrlib` |
| `core/models/pda/*`, `core/fluorescence/pda` | ⚪ tttrlib | built on `tttrlib.Pda`; models are 🟢 glue |
| `plugins/pch` (`compute_p1`, `pch_single_species`, `pch_open_system`, `pch_mixture`) | 🟡 MOVE | new framework staging `pch` |
| `core/models/rics/models.py` (`rics_simple`, `rics_diffusion_triplet`) | 🟡 MOVE | framework staging `fcs` / `rics` |

## Spectra & quenching

| ChiSurf path | Status | Destination |
|---|---|---|
| `plugins/spectra_downloader/*` | 🟢 STAY | DB downloaders; route spectra→R₀ through the consolidated `forster_radius_from_spectra` |
| `plugins/quenching_estimator` | 🟢 STAY | wraps `quest`'s dye-diffusion, which already uses the framework's AV |

# Cross-project notes
- **quest is already clean.** It imports the framework's AV directly (with a fallback labeling library), uses no `tttrlib`, imports ChiSurf only for a GUI viewer preference. No spectroscopy duplication with ChiSurf. Its one overlap is `quest/lib/tools/dye_diffusion` ↔ the framework staging `cgdye` (both coarse-grained dye simulation) — a separate consolidation, **out of scope here**.
- **The boundary matters.** Pushing burst/PDA/correlation into the framework would add a second `tttrlib` dependency inside a structure-modeling library. Keep them out.

# Target layout in the framework's Python staging layer
```
<framework>.decay        ← decay synthesis, spectrum binning, sensitized-acceptor, linearization-table calc
<framework>.irf          ← synthetic IRF, rising-edge detection, RL deconvolution
<framework>.fret         ← R0-from-spectra + overlap integral (merge chisurf forster.py with cgdye/r0.py)
<framework>.anisotropy   ← vm_rt_to_vv_vh, Perrin, G-factor
<framework>.kappa2       ← κ² distributions (merge chisurf kappa2.py with cgdye κ²)
<framework>.phasor       ← phasor / FLIM
<framework>.fcs          ← diffusion kernels, MEM 1D/2D, ILT, RICS, curve merge/normalize
<framework>.pch          ← photon-counting histogram
(<framework>.distributions / distance_metrics already exist — delete chisurf duplicates)
```

# Design

## Import seam in ChiSurf
Introduce a thin compatibility shim so the swap is one-line per call site and reversible:

```python
# chisurf/core/fluorescence/_backend.py
"""Single import seam for spectroscopy physics now owned by the modeling framework.

ChiSurf must not reimplement decay/FRET/anisotropy/distribution physics.
Import it from here; this module re-exports the framework's compiled core and
its Python staging-layer symbols under stable ChiSurf-facing names.
"""
# import the compiled modeling framework core + its Python staging layer
# (distributions, distance_metrics, ...)
# ... re-export convolve, rescale, fret_efficiency, R0, etc.
```

Call sites in `core/models/tcspc/*`, `core/fluorescence/*` import from `_backend` instead of from the deleted local modules. When the framework is unavailable (e.g. a stripped CI image), `_backend` raises a clear, actionable ImportError naming `pixi run build-extensions`.

## Numeric-equivalence guard (gate for every swap)
Each deletion/relocation is gated by a parametrized test that pins ChiSurf's old output against the framework output **before** the old code is removed:

```python
# test/spectroscopy/test_impbff_equivalence.py
@pytest.mark.parametrize("case", DECAY_CASES)
def test_convolution_matches_backend(case):
    old = _legacy_convolve(case.spectrum, case.irf, case.time)   # captured golden
    new = backend.convolve(case.spectrum, case.irf, case.time)
    np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-9)
```

Golden vectors are captured once from current ChiSurf, committed under `test/spectroscopy/golden/`, and the legacy implementation is deleted only after the framework path reproduces them.

# Migration path (task order)
1. **Inventory freeze** — capture golden vectors for every 🔴/🟡 module from the *current* ChiSurf into `test/spectroscopy/golden/`.
2. **🔴 DUP, phase 1 (zero new framework code needed):** delete and re-point imports for `distributions.py`, `convolve.py`, `tcspc.py:rescale_w_bg`, `fret/__init__.py`, `fret/forster.py`, and the `general.py` FRET/lifetime conversions. Gate each with the equivalence test. *This is the safe first PR.*
3. **🟡 MOVE, decay block:** land the framework staging `decay`, `irf`, `phasor` modules; re-point ChiSurf imports; delete locals.
4. **🟡 MOVE, anisotropy block:** land framework staging `anisotropy` + `kappa2` (merging `cgdye` κ²); re-point.
5. **🟡 MOVE, FCS block (largest):** land framework staging `fcs` (diffusion kernel, MEM 1D/2D, ILT, RICS, curve algebra); re-point `models/fcs`, `flc_2d`.
6. **🟡 MOVE, PCH:** land framework staging `pch`; re-point the plugin.
7. **Sweep:** `grep` ChiSurf for residual local physics; confirm `core/models/*` only *call* `_backend`, never reimplement.

# Acceptance criteria
- `grep -rn` over `chisurf/core/fluorescence` and `chisurf/core/math/functions` finds **no** local definition of convolution, decay rescaling, Förster radius, FRET-efficiency conversion, or the normal/generalized-normal/poisson_0toN PDFs; all resolve through `chisurf/core/fluorescence/_backend.py` to the framework / its staging layer.
- Every 🔴/🟡 module has a passing entry in `test/spectroscopy/test_impbff_equivalence.py` (rtol ≤ 1e-6).
- The framework's Python staging layer exposes the modules in *Target layout* above; each importable under the framework namespace with the namespace-package merge intact.
- `pixi run test` (and `test-doctest`) are green on a clean checkout after each phase.
- ⚪ burst / PDA / raw-correlation code is unchanged and still `tttrlib`-backed.
- A short note registering PRD-47 and its phase ordering is added to the roadmap index.

# Open questions
1. **Staging-layer release coupling** — ChiSurf will gain a hard runtime dependency on a recent version of the framework's Python staging layer. Pin a minimum version, or vendor the Python layer?
2. **MaxEnt / NNLS solvers** (`core/math/optimization/*`) — generic numerics that several plugins share. Move to the framework, or leave as ChiSurf-generic math?
3. **`quest` dye_diffusion ↔ `cgdye`** consolidation — spin up as a separate PRD?

# Relationships
- Independent refactor targeting the [Core target](/specs/core.md) fluorescence layer.
- Draws an explicit boundary: burst/PDA/raw-correlation stay with `tttrlib`, relevant to [PRD-50](prd-50.md) and [PRD-53](prd-53.md).
- Touches the fitting-model/AutoForm seam described in [GUI & AutoForm](/subsystems/gui-autoform.md) without changing it.
