# PRD-55 — Phasor Analysis Toolkit (PhasorPy Parity)

Status: Draft · Owner: ChiSurf core · Type: Feature
Related: PRD-40 (declarative dataset editors), PRD-38 (model/view-spec split),
PRD-28 (ndxplorer burst integration)

## 1. Motivation

[PhasorPy](https://www.phasorpy.org) (`github.com/phasorpy/phasorpy`, MIT) is an
open-source library for phasor-based FLIM and hyperspectral analysis. Its
distinctive value is a *headless phasor-algebra layer*, independent of acquisition
format (`phasor.py`, `lifetime.py`, `component.py`, `cursor.py`, `filter.py`).

ChiSurf already computes phasors well: `pixel_maps.phasor_maps()` (over
`tttrlib.CLSMImage.get_phasor`) produces calibrated per-pixel `g,s` maps, the
`img_pixel_phasor` plugin does IRF-reference calibration and photon thresholding,
and `phasor_section.py` renders the universal semicircle. **But ChiSurf does
nothing analytical with the phasor after plotting it.** A grep of the codebase
returns *zero* hits for each of the following standard phasor-FLIM operations:

1. **phasor → lifetime readout** — apparent phase lifetime τ_φ and modulation
   lifetime τ_M from `g,s`.
2. **median / gaussian filtering of `g,s` maps** — the near-universal phasor-FLIM
   denoising step (Digman & Gratton).
3. **component fraction / linear unmixing in phasor space** — reading species
   fractions off the line between two phasors; *the core idea of the phasor method*.
4. **cursor / ROI back-projection** — select a region on the phasor plot, highlight
   the corresponding image pixels.

ChiSurf is today a phasor **viewer**, not a phasor **analysis toolkit**. This PRD
closes that gap by reprogramming the four operations natively.

**Constraint — no new dependencies.** Every core PhasorPy dependency is already
shipped by ChiSurf (numpy, scipy, matplotlib, scikit-learn, scikit-image, tifffile,
pandas, numba). The toolkit is pure `numpy` + `scipy.ndimage` + `scipy.optimize` /
`numpy.linalg`. Nothing new is added to the environment; the PhasorPy sources are a
reference (cloned read-only into the git-ignored `thirdparty/phasorpy`), not a
dependency.

## 2. Scope

Delivered by **extending the existing dedicated phasor plugin**
`chisurf/plugins/microscopy/img_pixel_phasor/` in place — no new plugin scaffolding
and no new `chisurf/core` module. The pure-numpy analysis functions live in a new
Qt-free, tttrlib-free module `img_pixel_phasor/analysis.py` (operates on `g,s` map
arrays), consumed by the plugin's `core.py` / `view_model.py`.

### 2.1 The four operations (`img_pixel_phasor/analysis.py`)

- **`phasor_to_apparent_lifetime(g, s, frequency_mhz)` → `(tau_phi, tau_m)` (ns).**
  With ω = 2π·f:

  - τ_φ = ω⁻¹ · S / G
  - τ_M = ω⁻¹ · √(1/(G² + S²) − 1)

  `f` in MHz, scaled ×1e-3 so lifetimes come out in ns.
  Ref: PhasorPy `lifetime.phasor_to_apparent_lifetime`.

- **`phasor_filter_median(g, s, size=3, repeat=1)` → `(g, s)`** via
  `scipy.ndimage.median_filter` applied `repeat` times to each of `g` and `s`
  (NaN-safe; leaves the intensity/`mean` map untouched). Add a
  `phasor_filter_gaussian(g, s, sigma)` sibling using `scipy.ndimage.gaussian_filter`.
  Ref: PhasorPy `filter.phasor_filter_median` / `phasor_filter_gaussian`.

- **`phasor_component_fraction(g, s, c1, c2)` → fraction map** of component 1,
  projecting each pixel onto the line between two component phasors `c1=(g₁,s₁)`,
  `c2=(g₂,s₂)`, normalized and clipped to [0, 1]:

  f = [(g − g₂)(g₁ − g₂) + (s − s₂)(s₁ − s₂)] / [(g₁ − g₂)² + (s₁ − s₂)²]

  Plus **`phasor_unmix(g, s, components)`** for N ≥ 2 species via constrained least
  squares (`numpy.linalg.lstsq` for the unconstrained solve; `scipy.optimize.nnls`
  with a sum-to-one row for non-negative fractions), returning one fraction map per
  component. Ref: PhasorPy `component.phasor_component_fraction` / `phasor_component_fit`.

- **`mask_from_circular_cursor(g, s, center, radius)`** (+ `elliptic`, `polar`
  variants) → boolean mask with the image shape; **`pseudo_color(masks, colors)`** to
  colorize a stack of masks into an RGB label image. Kept as a lightweight
  *programmatic* primitive for headless/CLI/tests. **Interactive** cursor/ROI gating,
  clustering, and back-projection are delegated to ndXplorer (§4), not reimplemented.
  Ref: PhasorPy `cursor.py`.

### 2.2 Plugin surface (extend, don't rebuild)

- `core.py::compute_phasor()` — after computing `g,s`, attach derived maps
  (`tau_phi`, `tau_m`, and, when component endpoints are set, `fraction`) to the
  returned `maps` dict and to the imaging HDF5.
- `gui/view_model.py::PhasorImgViewModel` — add `tau_phi_map()`, `tau_m_map()`,
  `fraction_map()` accessors and state for filter size/repeat and the two component
  endpoints (`g₁,s₁,g₂,s₂`).
- `gui/phasor.view.json` — add `value` controls (filter size/repeat, component
  endpoints) and new `custom`/`image` docks for τ_φ, τ_M, and the fraction /
  pseudo-color map. No new AutoForm widget types are required.
- `gui/autoform/sections/phasor_section.py` — optional: a circular/elliptic
  pyqtgraph ROI on the phasor plot whose center/radius drives
  `mask_from_circular_cursor`, highlighting the selected pixels on the map docks
  (the lightweight in-plugin path; the full interactive story is ndXplorer, §4).

## 3. Reuse (exact paths)

- `chisurf/core/fluorescence/imaging/pixel_maps.py` — `phasor_maps`,
  `add_maps_to_hdf5` (produce the `g,s` the new module consumes).
- `chisurf/core/fluorescence/tcspc/phasor.py` — existing decay-phasor math
  (`Phasor`, `phasor_giw/siw`, FRET donor/DA/E); reference for semicircle/FRET
  geometry — do **not** duplicate.
- `chisurf/plugins/microscopy/img_pixel_phasor/{core,gui/view_model,gui/phasor.view.json}` —
  extension points above.
- `chisurf/gui/autoform/sections/phasor_section.py` — `@register_section("phasor")`
  pyqtgraph widget (semicircle + `g,s` density) to host the optional cursor ROI.
- No new dependencies: `numpy` + `scipy.ndimage` + `scipy.optimize`/`numpy.linalg`.

## 4. Integration decision — ChiSurf plugin vs. ndXplorer

The work splits along *phasor-specific domain math* vs. *generic multidimensional
exploration*, and the two halves go to different homes:

- **ChiSurf `img_pixel_phasor` plugin** owns the phasor-specific math (τ_φ/τ_M,
  `g,s` denoising, component fraction/unmixing, semicircle geometry). This must live
  in ChiSurf because it is TCSPC/fluorescence domain knowledge co-located with IRF
  calibration and `g,s` computation, and because **ndXplorer is a pluginless upstream
  submodule** (`modules/ndxplorer`, `fluorescence-tools/ndxplorer`) — patching it to
  add phasor semantics would be unmaintainable.
- **ndXplorer is reused, not reprogrammed, for the generic interactive layer.** It
  already provides 2-D density gating, cursor/ROI selection, clustering
  (HDBSCAN / K-means / UMAP), and back-projection to source rows, and its plugin
  already advertises "pixel-by-pixel analysis of multiparameter fluorescence …
  spatial correlation." Rather than reimplement cursors/clustering, ChiSurf routes
  the per-pixel phasor point-cloud DataFrame — columns `x, y, g, s, tau_phi, tau_m,
  fraction, intensity, n_photons`, already emitted as the standard imaging HDF5 —
  into ndXplorer via
  `chisurf/plugins/ndxplorer/mfdb_launcher.py::open_path_in_ndxplorer(path)`. Gating a
  region on the `g–s` density there already back-projects to the selected pixels.

Net: the plugin implements the four domain ops as pure functions (headless/CLI/tests
plus the plugin's own quick phasor plot); rich interactive segmentation and
clustering are delegated to ndXplorer by handing it the DataFrame.

## 5. Acceptance (headless — required)

Per the repo rule *every feature needs a headless test path*, tests live in
`chisurf/plugins/microscopy/img_pixel_phasor/test/` and run without a GUI:

- **Apparent lifetime.** A synthetic single-exponential decay of known τ places `g,s`
  on the universal semicircle at the expected coordinate, and
  `phasor_to_apparent_lifetime` returns τ_φ ≈ τ_M ≈ τ_true.
- **Fractions / unmixing.** A two-lifetime mixture of known fraction →
  `phasor_component_fraction` recovers the fraction within tolerance; `phasor_unmix`
  recovers ≥ 2-component fractions summing to 1.
- **Filtering.** `phasor_filter_median` reduces per-pixel `g,s` variance on a noisy
  constant-τ stack while preserving the mean coordinate.
- **Cursor.** `mask_from_circular_cursor` selects exactly the pixels whose `g,s` fall
  inside a known region.
- **CLI smoke.** Derived maps are produced through the plugin's `cli/main.py`.
- **ndXplorer round-trip (optional/GUI).** The phasor imaging HDF5 opens in ndXplorer
  and exposes the `g,s,tau_phi,…` columns for gating.

## 6. Non-goals

Deferred as niche (candidate for a later PRD): spectral / hyperspectral phasor and
spectral unmixing; absolute-concentration FLIM (`phasor_component_concentration`);
GMM phasor clustering; pawflim wavelet denoising; reprogrammed file readers (Leica
LIF, SimFCS `.ref`/`.R64`/`.B64`, FLIM-LABS JSON, OME-TIFF phasor, FBD); a standalone
matplotlib phasor-plot API. Interactive clustering/gating UI is **not** built here —
it is delegated to ndXplorer (§4).
