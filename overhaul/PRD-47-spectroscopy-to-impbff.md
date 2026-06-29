# PRD-47 — Relocate spectroscopy physics from ChiSurf into IMP.bff (via imp-tricks)

## Problem

The same fluorescence-spectroscopy physics is implemented in **three** repos:

- `imp.bff` — compiled C++/SWIG core (decay convolution, scoring, AV/FRET, photon
  statistics). The intended home.
- `imp-tricks` — a PEP-420 namespace-package mixin (`IMP.bff.*`) that adds Python
  enhancements on top of compiled IMP.bff *without* touching upstream IMP. The
  intended **staging ground** for new Python physics.
- `chisurf` — a desktop fitter that carries its **own** parallel copies of decay
  convolution, FRET conversions, distributions, Förster-radius math, etc.
- `quest` — a forward simulator that already consumes `IMP.bff.AV` correctly.

ChiSurf reimplements physics that IMP.bff / imp-tricks already provide (true
duplication), and also hosts physics that *should* live in IMP.bff but does not
yet (decay-side gaps). The result: two code paths drift, bug-fixes land in one
place only, and `quest`/`chisurf`/IMP all disagree on, e.g., the Förster-radius
prefactor.

**Goal of this PRD:** make IMP.bff (via imp-tricks for the Python layer) the
*single* home for spectroscopy physics; reduce ChiSurf to fitting-model glue,
GUI, and data-IO that *calls into* IMP.bff.

## Goals

1. **Delete every ChiSurf algorithm that already exists** in IMP.bff or
   imp-tricks `IMP.bff.*`; replace with an import (the 🔴 set below).
2. **Relocate ChiSurf-only physics** that belongs in IMP.bff into imp-tricks
   `IMP.bff.*`, then import it back (the 🟡 set).
3. **Keep fitting-model parameter groups, plugin GUIs, and data IO in ChiSurf**
   (the 🟢 set) — but have them delegate numerics to IMP.bff.
4. **Draw an explicit boundary**: photon-stream processing (burst search, PDA,
   raw correlation) stays with `tttrlib`, *not* IMP.bff (the ⚪ set).
5. **No behavioural regression** — ChiSurf's existing test suite stays green; a
   numeric-equivalence test guards each swap.

## Non-goals

- Rewriting IMP.bff's C++ core. New Python physics lands in imp-tricks first;
  promotion into compiled C++ is a later, separate decision.
- Moving burst detection / PDA / multi-tau correlation into IMP.bff (they are
  tttrlib-aligned single-molecule stream processing).
- Touching `quest`'s forward simulator beyond noting its overlap with
  `IMP.bff.cgdye` (tracked as a follow-up, not in this PRD).
- Changing ChiSurf's fitting framework, AutoForm, or plugin manifests.

## Legend

| Mark | Meaning | Action |
|------|---------|--------|
| 🔴 DUP | Already in IMP.bff / imp-tricks | Delete ChiSurf copy, import shared |
| 🟡 MOVE | Real physics, gap in IMP.bff | Relocate to imp-tricks `IMP.bff.*`, import back |
| 🟢 STAY | ChiSurf fitting/model/GUI glue | Keep; delegate numerics to IMP.bff |
| ⚪ tttrlib | Photon-stream processing | Leave with tttrlib; out of scope for IMP.bff |

## What the destination already provides

**IMP.bff (compiled C++/SWIG):** `DecayCurve`, `DecayConvolution` (6 kernels:
FAST/AVX/periodic/arbitrary-time-axis), `DecayLifetimeHandler`, `DecayScale`,
`DecayPileup` (Coates), `DecayPattern` (scatter), `DecayLinearization`,
`DecayScore` (7 χ² types), `DecayRoutines` (`decay_fconv*`, `decay_rescale*`,
`shift_lamp`, pile-up, discriminate), `PhotonStatistics`
(neyman/poisson/pearson/gauss/cnp/sswr + **polarized** wcm/twoIstar),
`AV` / `AVNetworkRestraint` / `PathMap` / `FPSReaderWriter`, and
`fret_efficiency` / `distance_fret` / `av_distance` / `av_distance_distribution`.

**imp-tricks `IMP.bff.*` (Python, namespace-merged):** `distance_metrics.py`
(FRET E↔R, asymmetric χ², AV-pair stats), `polymer.py` (Gaussian / worm-like-chain
linkers), `distributions.py` (normal, generalized-normal, poisson_0toN),
`fps.py` (fps.json IO), `av/` (BasicAV, ACV, numba kernels), `label/`,
`restraints/`, and the large `cgdye/` system (rotamer FRET, **R₀ from spectra**,
static/dynamic/dynamic+ regimes, exact kinetic master-equation, κ²).

## Scope — ChiSurf spectroscopy inventory & disposition

### TCSPC / fluorescence decay

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/tcspc/convolve.py` (`convolve_lifetime_spectrum_nb`, periodic, `convolve_decay_nb`) | 🔴 DUP | IMP.bff `DecayConvolution` / `decay_fconv*` |
| `core/fluorescence/tcspc/tcspc.py` → `rescale_w_bg` | 🔴 DUP | IMP.bff `DecayScale` / `decay_rescale_w_bg` |
| `core/fluorescence/tcspc/tcspc.py` → `bin_lifetime_spectrum`, `pddem` | 🟡 MOVE | `IMP.bff.decay` (spectrum helpers) |
| `core/fluorescence/general.py` (`fretrate_to_distance`, `distance↔fret_efficiency/rate`, `lifetime↔fret_efficiency`, `compute_mean_fret`, `combine_interleaved_spectra`) | 🔴 DUP | IMP.bff `fret_efficiency`/`distance_fret` + imp-tricks `distance_metrics.py` |
| `core/fluorescence/general.py` → `calculate_fluorescence_decay`, `fret_induced_donor_decay` | 🟡 MOVE | `IMP.bff.decay` (builds on `DecayConvolution`) |
| `core/fluorescence/tcspc/irf.py`, `irf_estimation.py` (synthetic IRF, rising-edge, RL deconvolution) | 🟡 MOVE | new `IMP.bff.irf` (its `generalized_normal_distribution` is 🔴 DUP of imp-tricks `distributions.py`) |
| `core/fluorescence/tcspc/corrections.py` (`compute_linearization_table`) | 🟡 MOVE | next to IMP.bff `DecayLinearization` (which applies, but can't compute, the table) |
| `core/fluorescence/tcspc/phasor.py` | 🟡 MOVE | new `IMP.bff.phasor` |
| `core/math/functions/distributions.py` (`poisson_0toN`, `normal_distribution`, `generalized_normal_distribution`) | 🔴 DUP | **exact** duplicate of imp-tricks `IMP.bff/distributions.py` |
| `core/models/tcspc/*` (lifetime, fret, anisotropy, mix_model, nusiance, av_decay, maxent, pddem, fret_structure) | 🟢 STAY | fitting-model parameter groups; back with IMP.bff kernels (the MaxEnt solver is a MOVE candidate) |

### FRET

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/fret/forster.py` (`overlap_integral`, `forster_radius`, `forster_radius_from_spectra`) | 🔴 DUP | consolidate with imp-tricks `cgdye/rotamer/r0.py` (already computes R₀ from spectra) |
| `core/fluorescence/fret/__init__.py` (`apparent_fret_efficiency`, `fret_efficiency`, `fret_efficency_to_fdfa`) | 🔴 DUP | IMP.bff `fret_efficiency` + imp-tricks `distance_metrics.py` |
| `core/fluorescence/fret/acceptor.py` (`da_a0_to_ad`) | 🟡 MOVE | `IMP.bff.fret` (uses convolution already in IMP.bff) |
| `core/fluorescence/fret/fret_line.py` + `plugins/fret_line` | 🟢 STAY | model-sweep generator; algorithm core could MOVE, GUI stays |

### Anisotropy

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/anisotropy/kappa2.py` (`kappasq*`, `kappasq_dwt`, `p_isotropic_orientation_factor`, `s2delta`) | 🟡 MOVE | consolidate with `cgdye` κ² → `IMP.bff.kappa2` |
| `core/fluorescence/anisotropy/decay.py` (`vm_rt_to_vv_vh`) | 🟡 MOVE | `IMP.bff.anisotropy` |
| `core/fluorescence/anisotropy/integrals.py` (Perrin anisotropy, G-factor) | 🟡 MOVE | `IMP.bff.anisotropy` |
| `plugins/jordi_anisotropy`, `jordi_g_factor`, `kappa2_dist` | 🟢 STAY | GUI wrappers over the MOVE targets above |

### FCS (largest physics gap in IMP.bff today)

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/fcs/correlate.py` (multi-tau, numba) | ⚪ tttrlib | prefer tttrlib's correlator over a second copy |
| `core/fluorescence/fcs/merge.py`, `normalization.py`, `filtered.py`, `channel_setups.py` | 🟡 MOVE | new `IMP.bff.fcs` (curve algebra) |
| `core/models/fcs/maxent.py` (`build_diffusion_kernel`, L-curve MEM) | 🟡 MOVE | `IMP.bff.fcs` |
| `plugins/fcs/flc_2d/fit/*` (ILT, MEM 1D/2D, kinetics) | 🟡 MOVE | `IMP.bff.fcs`; rest of `plugins/fcs/*` is GUI 🟢 |

### Burst / PDA / PCH / RICS

| ChiSurf path | Status | Destination |
|---|---|---|
| `core/fluorescence/burst/*` (filter, background, cusum, bocpd, kalman, bva) | ⚪ tttrlib | stays — stream processing |
| `plugins/burst/*` | 🟢 STAY | GUI + tttrlib |
| `core/models/pda/*`, `core/fluorescence/pda` | ⚪ tttrlib | built on `tttrlib.Pda`; models are 🟢 glue |
| `plugins/pch` (`compute_p1`, `pch_single_species`, `pch_open_system`, `pch_mixture`) | 🟡 MOVE | new `IMP.bff.pch` |
| `core/models/rics/models.py` (`rics_simple`, `rics_diffusion_triplet`) | 🟡 MOVE | `IMP.bff.fcs` / `IMP.bff.rics` |

### Spectra & quenching

| ChiSurf path | Status | Destination |
|---|---|---|
| `plugins/spectra_downloader/*` | 🟢 STAY | DB downloaders; route spectra→R₀ through the consolidated `forster_radius_from_spectra` |
| `plugins/quenching_estimator` | 🟢 STAY | wraps `quest`'s dye-diffusion, which already uses `IMP.bff.AV` |

## Cross-project notes

- **quest is already clean.** It imports `IMP.bff.AV` directly (LabelLib fallback),
  uses no tttrlib, imports ChiSurf only for a GUI viewer preference. No
  spectroscopy duplication with ChiSurf. Its one overlap is
  `quest/lib/tools/dye_diffusion` ↔ imp-tricks `IMP.bff.cgdye` (both coarse-grained
  dye simulation) — a separate consolidation, **out of scope here**.
- **The boundary matters.** Pushing burst/PDA/correlation into IMP.bff would add a
  second tttrlib dependency inside a structure-modeling library. Keep them out.

## Target layout in imp-tricks

```
IMP.bff.decay        ← decay synthesis, spectrum binning, sensitized-acceptor, linearization-table calc
IMP.bff.irf          ← synthetic IRF, rising-edge detection, RL deconvolution
IMP.bff.fret         ← R0-from-spectra + overlap integral (merge chisurf forster.py with cgdye/r0.py)
IMP.bff.anisotropy   ← vm_rt_to_vv_vh, Perrin, G-factor
IMP.bff.kappa2       ← κ² distributions (merge chisurf kappa2.py with cgdye κ²)
IMP.bff.phasor       ← phasor / FLIM
IMP.bff.fcs          ← diffusion kernels, MEM 1D/2D, ILT, RICS, curve merge/normalize
IMP.bff.pch          ← photon-counting histogram
(IMP.bff.distributions / distance_metrics already exist — delete chisurf duplicates)
```

## Design

### Import seam in ChiSurf

Introduce a thin compatibility shim so the swap is one-line per call site and
reversible:

```python
# chisurf/core/fluorescence/_backend.py
"""Single import seam for spectroscopy physics now owned by IMP.bff.

ChiSurf must not reimplement decay/FRET/anisotropy/distribution physics.
Import it from here; this module re-exports the IMP.bff (compiled) and
imp-tricks (IMP.bff.* Python) symbols under stable ChiSurf-facing names.
"""
import IMP.bff as bff                      # compiled core
from IMP.bff import distributions, distance_metrics   # imp-tricks Python layer
# ... re-export convolve, rescale, fret_efficiency, R0, etc.
```

Call sites in `core/models/tcspc/*`, `core/fluorescence/*` import from
`_backend` instead of from the deleted local modules. When IMP.bff is unavailable
(e.g. a stripped CI image), `_backend` raises a clear, actionable ImportError
naming `pixi run build-extensions`.

### Numeric-equivalence guard (gate for every swap)

Each deletion/relocation is gated by a parametrized test that pins ChiSurf's old
output against the IMP.bff output **before** the old code is removed:

```python
# test/spectroscopy/test_impbff_equivalence.py
@pytest.mark.parametrize("case", DECAY_CASES)
def test_convolution_matches_impbff(case):
    old = _legacy_convolve(case.spectrum, case.irf, case.time)   # captured golden
    new = backend.convolve(case.spectrum, case.irf, case.time)
    np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-9)
```

Golden vectors are captured once from current ChiSurf, committed under
`test/spectroscopy/golden/`, and the legacy implementation is deleted only after
the IMP.bff path reproduces them.

## Migration path (task order)

1. **Inventory freeze** — capture golden vectors for every 🔴/🟡 module from the
   *current* ChiSurf into `test/spectroscopy/golden/`.
2. **🔴 DUP, phase 1 (zero new IMP.bff code needed):** delete and re-point imports
   for `distributions.py`, `convolve.py`, `tcspc.py:rescale_w_bg`,
   `fret/__init__.py`, `fret/forster.py`, and the `general.py` FRET/lifetime
   conversions. Gate each with the equivalence test. *This is the safe first PR.*
3. **🟡 MOVE, decay block:** land `IMP.bff.decay`, `IMP.bff.irf`,
   `IMP.bff.phasor` in imp-tricks; re-point ChiSurf imports; delete locals.
4. **🟡 MOVE, anisotropy block:** land `IMP.bff.anisotropy` + `IMP.bff.kappa2`
   (merging `cgdye` κ²); re-point.
5. **🟡 MOVE, FCS block (largest):** land `IMP.bff.fcs` (diffusion kernel, MEM
   1D/2D, ILT, RICS, curve algebra); re-point `models/fcs`, `flc_2d`.
6. **🟡 MOVE, PCH:** land `IMP.bff.pch`; re-point the plugin.
7. **Sweep:** `grep` ChiSurf for residual local physics; confirm `core/models/*`
   only *call* `_backend`, never reimplement.

## Acceptance criteria

- `grep -rn` over `chisurf/core/fluorescence` and `chisurf/core/math/functions`
  finds **no** local definition of convolution, decay rescaling, Förster radius,
  FRET-efficiency conversion, or the normal/generalized-normal/poisson_0toN PDFs;
  all resolve through `chisurf/core/fluorescence/_backend.py` to IMP.bff /
  imp-tricks.
- Every 🔴/🟡 module has a passing entry in
  `test/spectroscopy/test_impbff_equivalence.py` (rtol ≤ 1e-6).
- imp-tricks exposes the modules in *Target layout* above; each importable as
  `IMP.bff.<name>` with the namespace-package merge intact.
- `pixi run test` (and `test-doctest`) are green on a clean checkout after each
  phase.
- ⚪ burst / PDA / raw-correlation code is unchanged and still tttrlib-backed.
- A short note is added to `overhaul/README.md` / `MASTER-ORDER.md` registering
  PRD-47 and its phase ordering.

## Open questions

1. **imp-tricks release coupling** — ChiSurf will gain a hard runtime dependency
   on a recent imp-tricks. Pin a minimum version, or vendor the Python layer?
2. **MaxEnt / NNLS solvers** (`core/math/optimization/*`) — generic numerics that
   several plugins share. Move to IMP.bff, or leave as ChiSurf-generic math?
3. **`quest` dye_diffusion ↔ `cgdye`** consolidation — spin up as PRD-48?
