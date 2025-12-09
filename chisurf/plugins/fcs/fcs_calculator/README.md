# FCS Confocal Calculator (ChiSurf)

This plugin provides an interactive **FCS confocal diffusion/volume calculator** inside ChiSurf. It is inspired by, and partially based on, the theory implemented in **QuickFit3**'s *Diffusion Coefficient Calculator* (`calc_diffcoeff`), but reimplemented in Python/PyQt5 and integrated with ChiSurf's workflow.

## Features

- Coupled FCS confocal parameters:
  - Correlation time: **τ** (µs)
  - Diffusion coefficient: **D** (µm²/s)
  - Hydrodynamic radius: **rₕ** (nm)
  - Effective confocal volume: **Veff** (fL)
  - Mean number of molecules: **N**
  - Concentration: **c** (nM)
- **Exclusive constraints** (radio buttons):
  - *Fix D* → compute `Veff` and `rₕ`
  - *Fix rₕ* → compute `D` and `Veff`
  - *Fix Veff* → compute `D` and `rₕ`
- Temperature / viscosity aware:
  - Water viscosity η(T) via empirical Kapusta-style relation
  - Optional override of η to handle non-aqueous or effective viscosities
- **Reference dye database** (Kapusta 2010, etc.):
  - D@25 °C (water) for common fluorophores and a few biomolecules
  - Optional D-scaling from 25 °C / water to arbitrary (T, η)
- **Molecular-shape support** (QuickFit3-inspired):
  - Sphere (Stokes–Einstein)
  - Ellipsoid (Perrin friction factor)
  - Cylinder (Hansen 2004 approximation)
- **Settings persistence**:
  - Export current state to JSON file
  - Import JSON to restore a previous calculator configuration

## Underlying Models

### 1. Water Viscosity Model

For water between ~0–100 °C the plugin uses the empirical relation

\[\eta(T) = A \cdot 10^{B/(T-C)}\]

with

- `T` absolute temperature (K)
- `A = 2.414·10⁻⁵ Pa·s`
- `B = 247.8 K`
- `C = 140 K`

This is the same functional form used in QuickFit3 (`calc_diffcoeff`) and in many FCS/viscosity references.

### 2. Stokes–Einstein (Sphere)

For a spherical particle of hydrodynamic radius `rₕ` in a solution of viscosity `η` at temperature `T`:

\[ D = \frac{k_B T}{6 \pi \eta r_h} \]

where `k_B` is Boltzmann's constant.

The inverse relation is used to compute `rₕ` from a known `D`.

### 3. Scaling from Reference D (D@25 °C, Water)

Given a reference diffusion coefficient `D₍₂₅,W₎` (at 25 °C in water, viscosity η₂₅,W), the plugin uses the classical scaling

\[ D(T, \eta) = D_{25,W} \cdot \frac{T}{298.15\,\mathrm{K}} \cdot \frac{\eta_{25,W}}{\eta(T)}. \]

This is directly analogous to the `D_{20,W}` scaling described in the QuickFit3 help for the diffusion calculator, simply with a different reference temperature.

### 4. Ellipsoid (Perrin Factor)

For a rotationally symmetric ellipsoid with diameters `a` (rotation axis) and `b` (other axes), the diffusion coefficient is

\[ D = \frac{k_B T}{6 \pi \eta R_e F_t}, \]

where

- Equivalent radius: \(R_e = (a b^2)^{1/3}\)
- Axial ratio: \(p = a/b,\ q = 1/p\)
- Translational Perrin factor `F_t` (Perrin 1934):
  - Prolate (p > 1):

    \[ F_t = \frac{\sqrt{1-q^2}}{q^{2/3} \ln\left(\frac{1+\sqrt{1-q^2}}{q}\right)} \]

  - Oblate (p < 1):

    \[ F_t = \frac{\sqrt{q^2-1}}{q^{2/3} \arctan\left(\sqrt{q^2-1}\right)} \]

The ChiSurf implementation follows the same formulas as the QuickFit3 documentation for `calc_diffcoeff`.

### 5. Cylinder (Hansen Approximation)

For a circular cylinder of diameter `d` and length `L` with aspect ratio `p = L/d`, the diffusion coefficient is again

\[ D = \frac{k_B T}{6 \pi \eta R_e F_t}, \]

with equivalent radius

\[ R_e = \left(\frac{3}{2 p^2}\right)^{1/3} \frac{L}{2} \]

and Hansen's polynomial approximation for the translational Perrin factor (Hansen 2004)

\[ F_t(p) = 1.0304 + 0.0193 x + 0.06229 x^2 + 0.00476 x^3 + 0.00166 x^4 + 2.66 \cdot 10^{-6} x^7, \quad x = \ln p. \]

This matches the formula given in the QuickFit3 `calc_diffcoeff` online help.

### 6. FCS Confocal Coupling

The widget couples these diffusion models to FCS confocal concepts:

- For a given `D` and structure factor `S = w_z / w_xy`, the effective confocal volume is

  \[ V_\text{eff} = \pi^{3/2} S (4 D \tau_D)^{3/2} \]

- The concentration `c` and number of molecules `N` in `Veff` are related via

  \[ N = N_A \cdot c_{\text{mol/L}} \cdot V_{\text{eff, m}^3} \approx 0.6022 \cdot c_{\text{nM}} \cdot V_{\text{eff, fL}}. \]

These standard relations are the same as those used in classical FCS analysis and are conceptually compatible with the QuickFit3 FCS plugins.

## JSON Settings

The calculator allows saving and loading its full UI state as a JSON file. The JSON contains, among others:

- `tau_us`, `D_um2_s`, `rh_nm`, `S`, `veff_fL`
- `temp_C`, `eta_mPa_s`, `use_water_eta`
- `conc_nM`, `num_mols`, `invN`
- Constraint mode: `fix_mode` (`"D"`, `"rh"`, or `"V"`)
- Dye selection and scaling (`dye`, `scale_dref`)
- Molecular shape settings (`shape_type`, `shape_size_nm`, `shape_aspect`)

This makes it easy to share or archive calculator configurations alongside analysis notebooks.

## References

Many of the formulas implemented here follow the QuickFit3 *Diffusion Coefficient Calculator* documentation and its cited references, including:

- J. W. Krieger, J. Langowski, **QuickFit3**, Diffusion Coefficient Calculator help (GPL software).
- F. Perrin (1934): *Mouvement brownien d'un ellipsoide*. J. Phys. Radium.
- S. Hansen (2004): *Translational friction coefficients for cylinders of arbitrary axial ratios*. J. Chem. Phys.
- T. M. Laue et al. (1992): *Computer-Aided Interpretation of Analytical Sedimentation Data for Proteins*.

QuickFit3 is licensed under GPL; this plugin reuses its theoretical framework but is implemented independently in Python within the ChiSurf ecosystem.
