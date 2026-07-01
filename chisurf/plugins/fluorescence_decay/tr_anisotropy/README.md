# Time-Resolved Anisotropy Wizard

Guided construction of a **linked VV/VH global anisotropy fit** from
polarisation-resolved fluorescence decays.

## Workflow (GUI wizard)

1. **Welcome** — overview.
2. **Data** — select the VV/VH IRF and sample-decay files.
3. **Normalize IRF** — drag a background region on the interactive plot; the VV/VH
   IRFs are background-subtracted and intensity-matched (30 %–80 % default region).
4. **Corrections** — set the g-factor and l1/l2 channel-mixing factors.
5. **Components** — define the lifetime and rotation spectra.
6. **Finish** — create the VV, VH and global fits with all parameters linked.

## Architecture (new plugin standard)

```
tr_anisotropy/
  manifest.json               plugin metadata + gui/cli entrypoints
  core/irf.py                 Qt-free IRF background subtraction + normalisation
  core/spectra.py             lifetime/rotation spectrum I/O (*.spk.json)
  core/fits.py                VV/VH parameter link/constraint plan (as data)
  gui/view_model.py           AnisotropyViewModel (state, loading, fit creation)
  gui/tool.py                 AutoForm host (AnisotropyWizard / ChisurfWizard alias)
  gui/irf_widget.py           embedded interactive IRF region selector (pyqtgraph)
  gui/components_widget.py    embedded lifetime/rotation spectrum tables
  anisotropy.view.json        declarative wizard layout (AutoForm)
  cli/main.py                 `anisotropy irf-correct` / `anisotropy spectrum`
  test/                       headless tests (no Qt)
```

The scientific numerics (IRF correction, spectrum persistence, the VV↔VH link
plan) live in the Qt-free `core/` package, so they are unit-tested without a GUI
and reused by the CLI. The genuinely interactive pieces (the draggable IRF
background region and the component tables) are small embedded Qt widgets.

**Fix vs. the legacy wizard:** the old component tables used opposite column
orders in their *add* buttons versus their *load* path, so manually added
lifetime/rotation components were silently swapped (amplitude ↔ value). Both
tables now use one consistent **amplitude, value** order, matching how the
spectra are stored and consumed.

## CLI

```bash
# background-correct and intensity-match a VV/VH IRF pair (headless)
csc anisotropy irf-correct --vv irf_vv.txt --vh irf_vh.txt --lb 300 --ub 800

# print the stored default lifetime/rotation spectra
csc anisotropy spectrum
```

## User settings

Spectra are stored under `<user_settings>/plugins/tr_anisotropy/wizard.spk.json`
(seeded from the packaged default on first use; a `.backup.json` is kept when the
default is overwritten). Instrument corrections are stored in
`<user_settings>/anisotropy_corrections.json`.

## Jordi format

When `cs.current_setup.is_jordi` is set, a single IRF file and a single data file
hold both polarisations; the loader reads each twice with the VV/VH polarisation
set accordingly (put the combined file in the VV fields).
