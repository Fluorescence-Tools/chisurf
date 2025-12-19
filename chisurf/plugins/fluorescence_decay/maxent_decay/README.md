# MaxEnt TCSPC lifetime / FRET plugin

This development plugin implements maximum-entropy (MaxEnt) analysis for
TCSPC decays in two modes:

- **Lifetime mode**: distribution of lifetimes `p(τ)`.
- **FRET distance mode**: distribution of donor–acceptor distances `p(R)`
  with an explicit donor lifetime spectrum (me\_vin4\_E and
  me\_vin4\_E\_do analogues).

The implementation is self-contained in this folder and does **not** modify
core `chisurf` models.

## Modules

- `__init__.py`
  - Plugin metadata and `load()` function returning `MaxentDecayWidget`.
- `fmem/`
  - `core.py`: numerical MaxEnt solvers (`solve_lifetime_mem`, `solve_fret_mem`).
  - `api.py`: convenience helpers for scripts/notebooks (grid builders and `run_*` helpers).
  - `gui.py`: Qt/pyqtgraph front-end (`MaxentDecayWidget`).
  - `settings.py`: JSON-based user defaults.
  - `sampling.py`: optional `emcee`-based sampling utilities.

## GUI usage

### 1. Opening the plugin

1. In ChiSurf, load or create a TCSPC **fit** (lifetime or FRET model).
2. Open the plugin via the menu:
   - `Plugins → Fluorescence decay → MaxEnt lifetime MEM`.

The plugin opens in its own window with a **control panel** on the left
and four plots (decay, residuals, distribution, L‑curve) on the right.

### 2. Common steps (both modes)

1. In the **Data** group, click **Refresh from current fit**.
   - This binds the plugin to `cs.current_fit` and copies the data,
     time axis, and fit range.
2. In the **IRF** group:
   - Leave as default to use `model.convolve.irf` from the current fit,
     or
   - Click **Select IRF dataset** to choose an imported IRF curve.
   - Click **Clear IRF selection** to fall back to the model IRF.
3. Adjust the **fit range** by dragging the grey region in the decay
   plot if needed.
4. Use the **nu (reg)** spin box and **L‑curve** button to scan
   different regularization strengths and automatically pick a
   reasonable `nu`.

### 3. Lifetime mode workflow

1. In **Mode**, select **Lifetime**.
2. Set the **tau grid [ns]** (min, max, step).
3. Optional: click **Load prior vector** to supply a custom prior over
   `tau`. The vector length must match the tau grid size.
4. Set the **start @ frac of peak** to control where the fit range
   starts (as fraction of the decay peak).
5. In the nuisance section:
   - **timeshift [channels]**, **background [cts]**, **IRF background [cts]**
     are initial values.
   - Check **Fit nuisance (ts/bg/IRF BG)** to let the plugin optimize
     these three parameters.
6. Click **Run MEM**.
   - The upper plot shows data and MEM fit.
   - The second plot shows weighted residuals.
   - The third plot shows the recovered lifetime distribution `p(τ)`.
7. If desired, click **Save result** to write distribution, decay/fit,
   IRF, residuals and a `meta.json` with all settings into a folder.

### 4. FRET distance mode workflow

1. In **Mode**, select **FRET**.
2. Set **tau0 [ns]** and **R0 [Å]** (Förster radius).
3. Set the excitation **Period [ns]**. When you refresh from a fit this
   is initialized from the decay time axis (acquisition window) but can
   be edited.
4. Set the **R/R0 range** (min, max, #points) for the distance grid.
5. Donor‑only spectrum:
   - Click **Load donor spectrum** to load `(amplitude, lifetime)`
     pairs from a text/CSV file, or
   - Click **Load donor from fit** to take the donor lifetime spectrum
     from an existing lifetime or FRET fit, or
   - Leave it empty to use the default single‑exponential donor with
     lifetime `tau0`.
6. Set **donor‑only fraction**: the fraction of donor‑only population
   (0–1). If **Fit nuisance** is enabled, this value becomes a fitted
   parameter.
7. Optional: click **Load distance prior** to supply a prior over the
   distance axis. The vector length must match the R grid size.
8. Nuisance / backgrounds:
   - Set **timeshift [channels]**, **background [cts]**, **IRF background
     [cts]**, and **lamp scatter**.
   - Check **Fit nuisance (ts/bg/IRF BG)** to let the plugin optimize
     timeshift, decay background, lamp scatter and the donor‑only
     fraction. IRF background is currently fixed by the spin box or by
     automatic tail estimation.
9. Click **Run MEM**.
   - The upper plot shows decay, MEM fit, and IRF.
   - The second plot shows weighted residuals.
   - The third plot switches to **Distance distribution** and displays
     `p(R)`.
10. Use **L‑curve** in FRET mode exactly as in lifetime mode to choose
    a regularization strength.
11. Use **Save result** to export distance distribution, decay/fit,
    IRF, residuals and `meta.json` for further analysis.

## Parameter conventions

- **`dt`**
  Time per detector channel (typically in ns).
- **`timeshift`**
  IRF shift in **detector channels (samples)** (fractional values allowed).
- **`fitrange`**
  `(start, stop)` indices in detector channels.
- **FRET distance grid**
  The solver uses `R` in Å, but the GUI defines the grid via **fractions of `R0`**:
  `R = linspace(r_min_frac * R0, r_max_frac * R0, r_bins)`.

## Reference

The underlying maximum-entropy algorithm is based on:

- V. Vinogradov and B. Wilson, "Quantitative decay analysis by the maximum
  entropy method", *Applied Spectroscopy* **54** (2000) 849–855.
