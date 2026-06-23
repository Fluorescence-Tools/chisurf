# PRD: Centralized Detector Setup Selection

## Problem

The `DetectorWizardPage` (a complex full-featured detector/PIE-window definition
widget) is instantiated in **15+ plugin UIs** across the codebase. Each plugin
embeds the entire wizard page — tables, TTTR reading controls, calibration
selectors, etc. — even when only a simple setup combo + detector selector would
suffice. This causes:

- **UI clutter** in plugin windows where detector setup is secondary to the
  plugin's main task.
- **Redundant complexity** — users must interact with the full wizard in every
  plugin rather than once centrally.
- **Maintenance burden** — any change to the detector wizard propagates to every
  embedding site.
- **Inconsistent UX** — each plugin exposes a slightly different subset of
  visibility flags.

## Goal

Replace the embedded `DetectorWizardPage` in plugin UIs with a lightweight
`SetupSelectorWidget` that provides just:

1. A combo box listing saved detector setups.
2. A **"Setup Editor…"** button that opens the full `DetectorWizardPage` in a
   modal dialog (centralised editing).
3. A **tooltip** on the combo that summarises the selected setup settings
   (windows, detectors, g-factors, micro-time ranges, TTTR reading params).

The central `DetectorWizardPage` remains available for:

- The `SetupChannelDefinitionWidget` settings tool (used in preferences).
- The boarding/onboarding wizard.
- The modal dialog launched from `SetupSelectorWidget`.

## New Component: `SetupSelectorWidget`

**File:** `chisurf/gui/widgets/wizard/tttr_channeldefinition/setup_selector.py`

### Public API

| Member | Type | Description |
|--------|------|-------------|
| `setupChanged` | `Signal()` | Emitted when the selected setup changes or the editor saves. |
| `__init__(parent=None)` | | Populates combo from `load_detector_setups()`. |
| `get_settings()` | `→ dict` | Returns full setup dict (windows, detectors, tttr_reading) for the selected setup. |
| `set_settings(data)` | | Sets internal setup data and updates combo. |
| `current_setup_name` | `str` property | Name of the currently selected setup. |
| `refresh()` | | Repopulate the combo from the detector setups file. |

### Internal Layout

```
┌──────────────────────────────────────────────┐
│ [Setup: _______________▼] [Setup Editor…]     │
│ Tooltip: "PR: windows=…, detectors=…"         │
└──────────────────────────────────────────────┘
```

- `QComboBox` — setup names from `detector_setups.json` / MFDB.
- `QToolButton` / `QPushButton` — "Setup Editor…".
- Tooltip dynamically generated from the selected setup data.
- A `QHBoxLayout` in a `QWidget` or `QGroupBox`.

### Behaviour

1. **Combo selection changed:** Emit `setupChanged`. Update tooltip.
2. **"Setup Editor…" clicked:** Open `DetectorWizardPage` in a `QDialog`. If the
   dialog is Accepted, read settings back from the page, store internally, emit
   `setupChanged`, update tooltip.
3. **Tooltip format:** Multi-line string:
   ```
   Setup: MySetup
   Windows: PIE (0:100), PR (100:200)
   Detectors: green (chs=0,1), red (chs=2,3)
   TTTR: SPC-130, 50ps, 50ps, bin=1
   ```

## Files to Modify (usage sites)

Priority order — **P0** first (simplest replacements), **P1** (need minor
adaptation), **P2** (wizard pages need more thought).

### P0 — Direct embed replacements

These plugins embed `DetectorWizardPage` as a tab or page in a stacked widget.
Replace with `SetupSelectorWidget` and remove the full page.

| File | Line(s) | Current pattern |
|------|---------|-----------------|
| `chisurf/plugins/tttr/tttr_count_rate_analysis/__init__.py` | 90-100 | Full page in "Channel Definition" tab |
| `chisurf/plugins/tttr/audifier/gui.py` | 219-220 | Full page in "Setup" tab |
| `chisurf/plugins/tttr/tttr_image_browser/__init__.py` | 107-118 | Full page in page 0 |
| `chisurf/plugins/tttr/trace_browser/__init__.py` | 424-445 | Full page in page 0 |

### P1 — Need layout adaptation

These embed the page similarly but have additional wiring or dependent widgets.

| File | Line(s) | Notes |
|------|---------|-------|
| `chisurf/plugins/burst/burst_mle_analysis/wizard.py` | 1408-1412 | In "Detector Definition" tab; wired to `_init_channels_from_wizard()` |
| `chisurf/plugins/burst/burst_background/__init__.py` | ~40+ | Embedded, needs state persistence port |
| `chisurf/plugins/burst/bid_to_analysis/__init__.py` | 598-611 | In "Setup" tab |
| `chisurf/plugins/microscopy/img_pixel_mle/imgmle.py` | 343-348 | In "Detector Definition" tab with detector combo |
| `chisurf/plugins/fcs/fcs_filter_calculator/gui_parts/main_window.py` | 61-74 | "Detector Setup" tab; keep `DetectorSelectionWidget` |
| `chisurf/plugins/fcs/fcs_2d/gui/wizard.py` | 110-121 | Lazy import in tab |
| `chisurf/plugins/tttr/microtime_histogram/wizard.py` | 489-503 | Already has simple combos; keep combos but replace page in tab |

### P2 — Wizard page replacements

These use `DetectorWizardPage` as the first page of a `QWizard`. Replace with
`SetupSelectorWidget` as a simple page; the full editor can be opened on demand.

| File | Line(s) | Notes |
|------|---------|-------|
| `chisurf/plugins/fcs/fcs_correlator/wizard.py` | 1008-1010 | First wizard page |
| `chisurf/plugins/tttr/photon_selection/wizard.py` | 25 | First wizard page |

### P3 — Already close to desired pattern

These already use a button-to-dialog pattern but still instantiate a hidden
`DetectorWizardPage`. Simplify to use `SetupSelectorWidget`.

| File | Line(s) | Current pattern |
|------|---------|-----------------|
| `chisurf/plugins/tttr/intensity_trace/__init__.py` | 679, 910-947 | Hidden page + `_open_setup_dialog()` |
| `chisurf/plugins/burst/burst_selection/gui/legacy/burst_selector.py` | 398-416 | Full page in a `QDialog` |

### Unchanged (remain as-is)

| File | Reason |
|------|--------|
| `chisurf/plugins/core/setup_channel_definition/gui/tool.py` | Centralised settings tool — needs full page |
| `chisurf/plugins/core/setup_channel_definition/wizard.py` | Full wizard for the channel definition |
| `chisurf/plugins/core/boarding/pages/detector_setups.py` | Onboarding — opens `DetectorWizard` directly |

## Implementation Steps

1. **Create `SetupSelectorWidget`** in
   `chisurf/gui/widgets/wizard/tttr_channeldefinition/setup_selector.py`.

2. **Export** from `chisurf/gui/widgets/wizard/tttr_channeldefinition/__init__.py`.

3. **For each P0/P1 usage site:**
   - Replace `DetectorWizardPage(...)` import + instantiation with
     `SetupSelectorWidget(...)`.
   - Remove any manual `load_detector_setups()` combo-population code.
   - Connect `setup_selector.setupChanged` to existing handler that reads
     `setup_selector.get_settings()` and processes the setup.
   - Remove old signal connections to `detector_wizard_page.*` widget methods.
   - If the host needed `detector_wizard_page.windows` / `.detectors` / `.filetype`
     properties, read them from `get_settings()` instead.

4. **For P2 (wizard pages):**
   - Create a thin `QWizardPage` subclass that wraps `SetupSelectorWidget`.
   - The page calls `setup_selector.get_settings()` in `initializePage()` to
     pre-populate downstream pages.

5. **For P3:**
   - Replace the hidden `DetectorWizardPage` with `SetupSelectorWidget`.
   - Keep the `_open_setup_dialog()` method but have it open the full editor
     dialog with the current settings pre-loaded.

6. **Update tests** in `test/gui/`:
   - `test_detector_wizard_page.py` — still valid (class unchanged).
   - Add `test_setup_selector_widget.py`.

7. **Verify** each plugin still works by running the application and testing
   setup selection flow.

## Non-Goals

- Removing `DetectorWizardPage` or `DetectorWizard` — these remain as the
  centralised full editors.
- Changing the detector setups file format or MFDB integration.
- Changing the `tttr_detector_setups.py` / `tttr_setup_utils.py` backend.
- Adding new plugin features unrelated to setup selection.
