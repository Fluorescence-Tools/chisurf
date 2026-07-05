# PRD-56 — ndXplorer ↔ ChiSurf RPC + Phasor Overlays

Status: Draft · Owner: ChiSurf core · Type: Feature
Related: **[PRD-55](PRD-55-phasor-analysis-toolkit.md)** (phasor analysis toolkit —
provides the phasor math this PRD exposes over RPC and consumes),
[PRD-28](PRD-28-ndxplorer-burst-integration.md) (ndXplorer ↔ MFDB burst integration),
[PRD-31](PRD-31-ndxplorer-headless-cli.md) (ndXplorer headless CLI),
[PRD-37](PRD-37-network-deployment-security.md) (network deployment security — gates
non-loopback exposure), [PRD-51](PRD-51-imaging-correlation.md) (imaging correlation).

## 1. Motivation

ndXplorer is ChiSurf's generic multidimensional explorer: today it reads files and
derives columns from local YAML equations (`settings/mfd.equations.yaml` →
`core/data_source.py::compute_values`), but it cannot tap ChiSurf's domain math.

[PRD-55](PRD-55-phasor-analysis-toolkit.md) builds a phasor analysis toolkit **inside**
ChiSurf (τ_φ/τ_M readout, `g,s` denoising, component fraction / unmixing, cursor
masks) and deliberately delegates the interactive gating/clustering layer to ndXplorer
— but that handoff is a **one-way file drop** (`open_path_in_ndxplorer(path)`). ndXplorer
receives a static DataFrame and does nothing phasor-aware with it: it draws no
semicircle, cannot read apparent lifetimes off `g,s`, and cannot ask ChiSurf to
recompute anything.

This PRD makes ndXplorer a **live phasor front-end**. It connects to ChiSurf over RPC
to compute apparent lifetimes, filter/unmix phasors, and fetch phasor reference
geometry, then draws the results with its own overlay and derived-column machinery.
The phasor math stays in ChiSurf (single source of truth, co-located with IRF
calibration and `g,s` computation); ndXplorer stays a **chisurf-free upstream
submodule** (`modules/ndxplorer`, `fluorescence-tools/ndxplorer`) and gains a
**first-class, reusable RPC client** — not a one-off patch.

**Directive:** improve ndXplorer into a proper ChiSurf RPC client (a clean,
documented, tested subsystem that speaks a transport contract), not an ad-hoc bolt-on.

## 2. Architecture — two halves and a clean seam

### 2.1 ndXplorer: first-class ChiSurf RPC client (chisurf-free)

A new `modules/ndxplorer/ndxplorer/rpc/` subsystem:

- **`RpcClient` interface (Protocol)** — the canonical `call(method, params) -> dict`
  shape (identical to ChiSurf's `ChisurfClient` / `InProcessClient`), plus connection
  lifecycle (connect / health / timeout / reconnect) and graceful no-server
  degradation.
- **`ZmqRpcClient`** — a small pyzmq + json JSON-RPC 2.0 REQ client that **absorbs and
  replaces** the ad-hoc `StandaloneZmqClient` / raw `zmq_cmd_port` wiring currently in
  `core/plot_main.py`. Used for headless / external connections.
- **Injected-client path** — `NDXplorer(chisurf_rpc=<obj>)` accepts *any* object
  satisfying the `RpcClient` interface, so the in-process GUI can inject ChiSurf's
  `InProcessClient` (no socket hop; ndXplorer runs in-process with the ChiSurf GUI)
  while headless injects a `ZmqRpcClient`.
- **`PhasorService`** — a typed facade over `client.call("phasor.*", …)` so call sites
  are readable and mockable.
- **Chisurf-free** — depends only on pyzmq + stdlib; it *defines* the transport
  contract and never imports chisurf. When no client is configured, ndXplorer behaves
  exactly as it does today.

Config surfaces: constructor `chisurf_rpc=` / legacy `zmq_cmd_port=` (GUI-injected),
CLI `--chisurf-rpc host:port`, env fallback (`CHISURF_RPC`).

### 2.2 ChiSurf: land the phasor math and expose it over `phasor.*`

- **`chisurf/plugins/microscopy/img_pixel_phasor/analysis.py`** — the
  [PRD-55](PRD-55-phasor-analysis-toolkit.md) subset, pure `numpy`/`scipy`, Qt-free and
  tttrlib-free (operates on `g,s` arrays):
  - `phasor_to_apparent_lifetime(g, s, frequency_mhz)` → `(tau_phi, tau_m)`
  - `phasor_filter_median(g, s, size=3, repeat=1)` / `phasor_filter_gaussian(g, s, sigma)`
  - `phasor_component_fraction(g, s, c1, c2)` / `phasor_unmix(g, s, components)`
  - `mask_from_circular_cursor(g, s, center, radius)` (+ elliptic) / `pseudo_color(masks, colors)`
  - **overlay-geometry helpers**: `universal_semicircle_polyline(harmonic=1)`,
    `iso_lifetime_contours(frequency_mhz, taus)` (τ_φ & τ_M grid),
    `lifetime_tick_markers(frequency_mhz, taus)`,
    `fret_trajectory(frequency_mhz, tau_d0, e_range)` — reusing
    `core/fluorescence/tcspc/phasor.py` geometry (do not duplicate).
  - Reference `thirdparty/phasorpy` **read-only**; no `phasorpy` import, no new
    dependency.
- **`chisurf/plugins/microscopy/img_pixel_phasor/backend/services.py`** — an RPC
  service registered via the plugin `manifest.json` `entrypoints.services`, mirroring
  `chisurf/plugins/pch/backend/services.py`. Namespace `phasor.*`, JSON-friendly
  arrays (lists) in and out:
  - `phasor.describe` → capabilities, default frequency / harmonic.
  - `phasor.apparent_lifetime` (g, s, freq → τ_φ, τ_M).
  - `phasor.filter` (g, s, kind, size/sigma, repeat → g', s').
  - `phasor.component_fraction` (g, s, c1, c2 → fraction) / `phasor.unmix`
    (g, s, components → fractions[]).
  - `phasor.cursor_mask` (g, s, center, radius/axes, kind → mask) /
    `phasor.pseudo_color` (masks, colors → RGB).
  - `phasor.overlays` (freq, harmonic, which set, optional component endpoints / FRET
    params → list of labelled polylines `{name, x[], y[], style}`).

  Register in `register_services(dispatcher)`; document the methods alongside the
  existing catalogue (`chisurf/server/protocol.py` / `server_methods.json`).

### 2.3 The seam — what flows over RPC

ndXplorer sends its current `g,s` column arrays → ChiSurf returns derived columns
(τ_φ, τ_M, fractions, filtered g,s), cursor masks / pseudo-color RGB, and overlay
polylines → ndXplorer injects the columns into its `DataSource` and draws the
polylines through `CurveOverlayWidget` as a new **server-fed overlay curve type**
(geometry computed remotely, not from a local YAML formula).

### 2.4 Shared overlay-**lines** interface (phasor lines + FRET lines)

Phasor reference geometry and smFRET FRET lines are both *parametric overlay lines on a
2-D plot*, so they share **one contract** — a **LineSet**: a list of
`{"name", "kind": "curve"|"scatter", "x": [...], "y": [...], "style": {...}, "axes": {...}}`.
Two ChiSurf providers return exactly this shape:

- `phasor.overlays` — semicircle, iso-lifetime grid/ticks, FRET trajectory, mixing line
  (from `analysis.build_overlays`, the single source shared with the imaging plugin and
  the phasor calculator).
- `fret_line.overlays` — static/dynamic/WLC/mixture FRET lines, reshaped from
  `fret_line.core.algorithms.compute_fret_line` (projections: E vs τ_f, E vs τ_x,
  τ_x vs τ_f), with an `axes` hint naming the x/y quantities.

On the ndXplorer side these are consumed through one uniform interface
(`ndxplorer/rpc/lines.py`): an `OverlayProvider` base with `PhasorLines` and `FretLines`
subclasses and a `LinesService` aggregating both, so ndXplorer draws phasor lines and
FRET lines through the same `CurveOverlayWidget` path. `NDXplorer` exposes both
`phasor_service` (the full toolkit) and `lines_service` (the uniform lines interface).

## 3. Feature set (all four)

1. **Reference geometry** — universal semicircle + iso-τ_φ/τ_M lifetime grid +
   mono-exponential lifetime markers, from `phasor.overlays`; drawn when the current
   2-D plot axes are `(g, s)`.
2. **Derived columns** — `phasor.apparent_lifetime` / `phasor.filter` results added as
   `tau_phi`, `tau_m`, `g_filt`, `s_filt` DataSource columns, plottable/gateable like
   any parameter.
3. **FRET + component lines** — `phasor.overlays(fret=…)` donor→DA quenching
   trajectory; two cursor-picked phasors → mixing line + `phasor.component_fraction` /
   `phasor.unmix` fraction column(s).
4. **Cursor pseudo-color** — gate a circular/elliptic region on the `g–s` density
   (ndXplorer's existing cursor/gating) → `phasor.cursor_mask` + `phasor.pseudo_color`
   → colorize the corresponding source rows/pixels.

### 3a. Additional ported PhasorPy plot geometry

Beyond the original four, the overlay toolkit (`analysis.build_overlays` /
`phasor.overlays`, and the standalone `phasor.contours`) now also produces the
PhasorPy-style geometry below, all as the same LineSet contract so they draw
unchanged in the calculator, the imaging plugin and ndXplorer:

- **`polar_grid`** — concentric circles + angular spokes with the unit circle flagged
  `major` (`polar_grid_polylines`); for reading phase and modulation.
- **`components`** — the N-component **mixing polygon** (no fractions) or
  fraction-weighted **mixing lines + mixture marker** (`component_mixing`), generalizing
  the two-component line to arbitrary component counts.
- **`cursor`** — circular / elliptic gating-cursor outlines (`cursor_polyline`), the
  visual companion to `phasor.cursor_mask`.
- **`phasor.contours`** — iso-density contour polylines of a 2-D phasor histogram
  (`density_contours`, via `contourpy`), for rendering dense clouds as contours.

## 4. ndXplorer UI / integration

A **"ChiSurf Phasor" toolbar** (`ndxplorer/ui/phasor_toolbar.py`, installed by
`plot_main.py` only when `phasor_service` is present) drives the chisurf-free helpers in
`ndxplorer/phasor_integration.py`: a frequency field plus **◐ Phasor overlays** (toggle —
draws the semicircle / iso-lifetime grid / ticks when the axes are `(g, s)`),
**📈 FRET line** (overlays a static Gaussian FRET line via `lines_service.fret_line`),
**τ φ/M** (computes τ_φ / τ_M columns from the g,s axes and injects them into the
`DataSource`), and **✕ Clear**. Overlay data coordinates are mapped through
`ndx.value_to_bin` against the current histogram edges, so LineSets land in the same
bin-space as the built-in curve overlays; both phasor lines and FRET lines use one draw
path. Every action fails soft (status message, never a crash), and the toolbar is absent
when no server is configured.

## 4a. Calculators hub — phasor plot (AutoForm + JSON)

The **phasor plot is also a standalone calculator** in the Calculators hub
(`chisurf/plugins/calculator/hub`), alongside the existing FRET-line and FRET/homoFRET
calculators. `chisurf/plugins/calculator/phasor_calculator` is a data-free, AutoForm
tool driven by a declarative `gui/phasor.view.json`: value/toggle controls (frequency,
harmonic, reference lifetimes, FRET donor lifetime, two component endpoints) beside the
`phasor` AutoForm section, which now draws **overlay polylines/markers** in addition to
the density + semicircle. Its model builds the overlays through the same
`analysis.build_overlays` used by `phasor.overlays`, so the calculator, the imaging
plugin and ndXplorer all render identical geometry. The FRET-line calculator and the
phasor calculator thus share the **LineSet** interface (§2.4): both are "line
generators" that emit the same overlay shape, one on the phasor plane and one on the
E-vs-τ plane.

## 5. Reuse (exact paths)

- **ChiSurf**: `chisurf/plugins/microscopy/img_pixel_phasor/{core.py, gui/view_model.py,
  cli/main.py, manifest.json}`; `chisurf/core/fluorescence/imaging/pixel_maps.py`
  (`phasor_maps`, `maps_to_dataframe`, `add_maps_to_hdf5`);
  `chisurf/core/fluorescence/tcspc/phasor.py` (semicircle/FRET geometry, `phasor_giw/siw`);
  `thirdparty/phasorpy` (reference only); `chisurf/server/dispatcher.py` +
  `chisurf/server/server_methods.json`; **pattern**:
  `chisurf/plugins/pch/backend/services.py` + its manifest `entrypoints.services`.
- **ndXplorer**: `core/plot_main.py` (`NDXplorer`, `_deferred_init`, existing zmq
  wiring ~L917–938); `core/data_source.py` + `core/data/data_manager.py` (column
  injection); `plotting/curve_overlay.py` + `CurveEvaluator`; `settings_helpers.py`;
  `cli.py`; `__main__.py::open_path_like_drop`.
- **FRET lines**: `chisurf/plugins/fret_line/core/algorithms.py`
  (`compute_fret_line`, new `fret_line_overlays`) + `backend/services.py`
  (`fret_line.overlays`, `fret_line.list_projections`).
- **Calculators hub**: `chisurf/plugins/calculator/hub/core/registry.py`
  (register the phasor entry), `chisurf/plugins/calculator/phasor_calculator/**`
  (new AutoForm calculator), `chisurf/gui/autoform/sections/phasor_section.py`
  (extended to draw overlays).
- **ndXplorer lines**: `modules/ndxplorer/ndxplorer/rpc/lines.py`
  (`OverlayProvider` / `PhasorLines` / `FretLines` / `LinesService`).
- **Glue**: `chisurf/plugins/ndxplorer/mfdb_launcher.py::open_path_in_ndxplorer`
  (inject client / port), `chisurf/plugins/ndxplorer/cli.py`.
- **No new dependencies**; no `phasorpy` import.

## 6. Connection model (both)

- **GUI (zero-config)** — the phasor plugin / ndXplorer launcher ensures a loopback
  ChiSurf RPC endpoint (reuse the in-process `ServiceDispatcher` via `InProcessClient`,
  or spawn `python -m chisurf.server`) and injects it into `NDXplorer(chisurf_rpc=…)`.
- **Headless / CLI** — `ndxplorer … --chisurf-rpc 127.0.0.1:8765`; plus a chisurf-side
  `csc` / `csg_*` wrapper.

Loopback-only per [PRD-37](PRD-37-network-deployment-security.md); non-loopback
exposure is gated on PRD-37 (CURVE/ZAP transport security + per-call authn/authz)
landing. Do not relax the loopback guard here.

## 7. Acceptance (headless — required)

Per the repo rule *every feature needs a headless test path*:

- **`analysis.py` units** ([PRD-55](PRD-55-phasor-analysis-toolkit.md) §5 subset):
  synthetic single-exponential `g,s` on the universal semicircle → τ_φ ≈ τ_M ≈ τ_true;
  two-lifetime mixture → `phasor_component_fraction` recovers the fraction within
  tolerance; `phasor_unmix` returns ≥ 2 fractions summing to 1; `phasor_filter_median`
  reduces per-pixel `g,s` variance while preserving the mean coordinate;
  `mask_from_circular_cursor` selects exactly the in-region pixels.
- **RPC contract**: an in-process `ServiceDispatcher` → each `phasor.*` method returns
  correct shapes and round-trips array payloads.
- **ndXplorer RPC client** (chisurf-free, mock server): connect / timeout / reconnect;
  the `RpcClient` interface; the `PhasorService` facade; column injection into
  `DataSource`; server-fed overlay polyline drawing (offscreen).
- **End-to-end headless**: start a loopback ChiSurf server (or inject an in-process
  client), run ndXplorer headless on a synthetic `g,s` CSV → the `tau_phi` column
  materializes and the semicircle overlay geometry is fetched.
- **GUI smoke (offscreen)**: the phasor panel appears on `(g,s)` axes; overlays render.

## 8. Non-goals

- Network transport security / TLS / per-call authz — deferred to
  [PRD-37](PRD-37-network-deployment-security.md).
- Any new ChiSurf dependency or a `phasorpy` import.
- Spectral / hyperspectral phasor and spectral unmixing (a
  [PRD-55](PRD-55-phasor-analysis-toolkit.md) non-goal).
- **Reimplementing phasor math inside ndXplorer** — it stays in ChiSurf, reached over
  RPC. ndXplorer only orchestrates, injects columns, and draws.
