# MaxEnt TCSPC lifetime / FRET plugin

This development plugin implements maximum-entropy (MaxEnt) analysis for
TCSPC decays in two modes:

- **Lifetime mode**: distribution of lifetimes `p(τ)` (mem\_vin4 analogue).
- **FRET distance mode**: distribution of donor–acceptor distances `p(R)`
  with an explicit donor lifetime spectrum (me\_vin4\_E and
  me\_vin4\_E\_do analogues).

The implementation is self-contained in this folder and does **not** modify
core `chisurf` models.

## Modules

- `core.py`
  - `mem_vin4_lifetime`: lifetime MaxEnt solver.
  - `mem_vin4_fret`: FRET distance MaxEnt solver, supporting
    donor-only spectra and lamp scatter.
- `gui.py`
  - `MaxentDecayWidget`: GUI for lifetime MEM on the current
    `cs.current_fit` dataset (pyqtgraph plots, IRF selector,
    prior loading, nuisance fitting switch).
- `cli.py`
  - Click-based CLI entrypoint used by `cli_entrypoint` from
    `__init__.py`.
- `__init__.py`
  - Plugin metadata and `load()` function returning `MaxentDecayWidget`.

## CLI usage

The plugin exposes a subcommand via the main `csc` CLI (see
`chisurf/cli.py`). Example invocations assume the plugin has been
installed and discovered as e.g. `maxent-decay`:

### Lifetime mode (default)

```bash
csc maxent-decay \
  --decay Decay_577D+577A+GTPgS.txt \
  --irf   Prompt.txt \
  --dt 0.0141 \
  --nu 1e-3 \
  --tau-min 0.001 --tau-max 10.0 --tau-step 0.02 \
  --fit-nuisance \
  --fit-start-fraction 0.9
```

Important options:

- `--fit-nuisance/--no-fit-nuisance`: enable/disable internal
  optimization of timeshift, decay background, and IRF background.
- `--fit-start-fraction`: start of the auto fit range as fraction of
  the peak (chisurf-like `initial_fit_range`).
- `--prior`: external prior vector (must match the tau grid length).
- `--lamp-background`: explicitly set the IRF background level; if
  omitted, it is estimated from the IRF tail.

### FRET distance mode

```bash
csc maxent-decay \
  --mode fret \
  --decay Decay_FRET.txt \
  --irf   Prompt.txt \
  --dt 0.0141 \
  --nu 5e-2 \
  --r-min 18.0 --r-max 180.0 --r-step 0.5 \
  --tau0 4.1 --r0 52.0 \
  --donor-spectrum donor_only.txt \
  --lamp-scatter 0.001 \
  --fit-start-fraction 0.9
```

In FRET mode the solver follows the Matlab scripts `me_vin4_E.m` and
`me_vin4_E_do.m`:

- `R` grid is defined by `--r-min`, `--r-max`, `--r-step` (Å).
- `--tau0` is the donor lifetime in absence of FRET (ns).
- `--r0` is the Förster radius `R0` (Å).
- `--lamp-scatter` scales the lamp curve added to each basis column.
- `--donor-spectrum` provides a donor-only spectrum as two columns
  (amplitude, lifetime). The file is flattened into
  `[a1, tau1, a2, tau2, ...]`.
- If `--donor-spectrum` is **omitted**, a single-component donor-only
  spectrum `[1, tau0]` is used.
- `--fit-nuisance` is **not** supported in FRET mode (nuisance
  optimization is lifetime-only).

The donor-only case of `me_vin4_E_do.m` is supported by providing a
multi-exponential donor spectrum via `--donor-spectrum` and choosing an
appropriate `--lamp-scatter` and `--fitrange` (implicitly controlled by
`--fit-start-fraction`).

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
4. Set the **RDA range [Å]** (min, max, #points) for the distance grid.
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

## Reference

The underlying maximum-entropy algorithm is based on:

- V. Vinogradov and B. Wilson, "Quantitative decay analysis by the maximum
  entropy method", *Applied Spectroscopy* **54** (2000) 849–855.
