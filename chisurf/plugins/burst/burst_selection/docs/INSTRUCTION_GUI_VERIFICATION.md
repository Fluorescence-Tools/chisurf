# Instruction: Desktop GUI Verification

## Goal

Validate the migrated PyQt GUI (`gui/tool.py`) against the legacy GUI
(`gui/legacy/burst_selector.py`) in a real desktop environment with Qt
rendering.

This is a **manual** procedure — no existing test can exercise
`QApplication` widgets headlessly on this system.

## Background

The migrated GUI has been incrementally fixed and passes all automated
tests, but has **never been opened in a desktop session**. Several
classes of issue can only be caught visually:

- Plot rendering (pyqtgraph)
- Layout overflow / clipping
- Missing signal connections
- Button/text visibility
- Drag-and-drop behaviour
- Progress dialog appearance

## Procedure

### 1. Launch the migrated GUI

```bash
cd /Users/tpeulen/dev/chisurf
python -m chisurf
```

From the menu bar:
**Spectroscopy > Single-Molecule > Burst-Selection**

### 2. Visual checklist

Compare against the legacy GUI (launched by setting
`USE_LEGACY_GUI = True` in `__init__.py` and restarting).

| Check | Expected | Legacy | Migrated |
|-------|----------|--------|----------|
| Window title | "Burst Selection" | ✓ | ? |
| Action bar buttons | Add TTTR, Batch, Process, Save .bur, Clear | ✓ | ? |
| Clear button visibility | Hidden (wizard passes `show_clear_button=False`) | ✓ | ? |
| File list area | Drag/drop hint, file paths appear | ✓ | ? |
| Output format checkboxes | CSV, MFD-HDF, Zip, Remove Folder | ✓ | ? |
| Filter group | Channels + microtime fields visible | ✓ | ? |
| dT min/max spinboxes | With enable checkboxes | ✓ | ? |
| Gap fill checkbox + spin | Functional | ✓ | ? |
| Filter mode combo | Burst / Count rate | ✓ | ? |
| Invert checkbox | Toggles enabled state | ✓ | ? |
| Threshold / TW / CR TW #Ph | Present and editable | ✓ | ? |
| Histogram group | Feature combo, bins, range, Auto, GMM controls | ✓ | ? |
| Plot group | Trace range, bin-widths, plot toggles, Refresh | ✓ | ? |
| GMM summary QTextEdit | Read-only, shows GMM results | ✓ | ? |

### 3. Functional test

#### 3a. Load a TTTR file

1. Drag `tests/data/bh_spc132_sm_dna/m000.spc` onto the file list
2. Click **Process**
3. **Expected:** Progress dialog appears, analysis completes, "Analysis complete" shown

#### 3b. Check dock tabs

After analysis:

| Tab | Expected content |
|-----|-----------------|
| Files | File list, output format checkboxes |
| Bursts | Table with burst data (Number of Photons, Duration, Proximity Ratio…) |
| Histogram | Histogram plot of selected feature |
| dT | Macro time difference plot |
| Filter | Photon filter indicator plot |
| MCS | Multichannel scaler trace |
| Decay | Fluorescence decay plot |
| Burst length | Burst duration histogram |

#### 3c. Save .bur

1. Click **Save .bur**
2. Choose a location
3. **Expected:** `.bur` file created, readable with the API:
   ```python
   from chisurf.plugins.burst.burst_selection.api.io import read_bur
   df = read_bur("/path/to/saved.bur")
   print(df.head())
   ```

#### 3d. Histogram + GMM

1. Select "Proximity Ratio" in the feature combo
2. Click **Fit GMM**
3. **Expected:** GMM plot overlay, component parameters in the summary text area
4. Verify: this should NOT auto-run on refresh (no `_fit_gmm_on_update`)

#### 3e. Diagnostic plots

1. Click **Refresh plots**
2. **Expected:** MCS, Decay, Burst length, Filter, dT tabs show plots

### 4. Edge cases

- **Empty state:** Open GUI with no files loaded — no crash
- **Batch dialog:** Click **Batch** — folder selection dialog opens
- **Clear:** Click **Clear** — files and results removed
- **Invalid channels:** Enter "abc" in channels — no crash, clear error message

### 5. Reproduce the `appendPlainText` scenario

1. Load a corrupt or minimal TTTR file that will raise during plot loading
2. **Expected:** Error message in summary QTextEdit (no crash)

## Success criteria

- All items in the visual and functional checklists pass
- No crashes, no unhandled exceptions in the console
- `.bur` output from migrated GUI is byte-identical to legacy output for the
  same input

## If something fails

Fix the issue directly in `gui/tool.py` and re-test. Do NOT fall back to
`USE_LEGACY_GUI = True` except for comparison — the goal is to identify and
fix all migrated GUI issues.

## After verification

Update `STATUS.md`:

- Mark all checklist items as ✅ or ❌
- Note any fixes that were applied
- If fully passing, advance to `INSTRUCTION_TTTRLIB_BURSTFILTER.md`
