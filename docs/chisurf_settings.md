# ChiSurf Global Settings

This document describes the main user-configurable settings used by ChiSurf.
Values are loaded from YAML files in `chisurf/settings` into `chisurf.settings`
(e.g. `chisurf.settings.cs_settings`, `chisurf.settings.gui`, `chisurf.settings.fret`).

Settings are grouped and ordered as in the YAML files:

- `settings_chisurf.yaml` – core application and analysis settings
- `experiment_configs.yaml` – experiment type registry and readers/models
- `settings_colors.yaml` – predefined color palette used by plotting code

Only fields that are actually read or clearly intended for use are documented
here.

---

## 1. `settings_chisurf.yaml`

### 1.1 `correlator` (TTTR-based FCS correlation)

Used by:
- `chisurf.plugins.tttr.tttr_correlate.gui.CorrelatorWidget` / `Correlator`
- `chisurf.gui.widgets.wizard.tttr_correlator.WizardTTTRCorrelator`
- `chisurf.plugins.burst_fcs_correlator`

Keys:

- **`B`**  
  Number of correlation lag bins *per cascade level* in the multi-tau
  algorithm used by `tttrlib.Correlator` (`n_bins` in the C++ code).
  For each coarsening level, `B` correlation points are computed before the
  time axis is coarsened again.

- **`number_of_cascades`**  
  Number of coarsening steps / cascade levels in the multi-tau correlator
  (`n_casc` in the C++ code). Each additional cascade halves the effective
  time resolution and adds another block of `B` correlation points at longer
  lag times, extending the accessible correlation window.

- **`split`**  
  Number of chunks the photon stream is split into for separate correlations.
  Used to compute multiple partial correlations that are then averaged.

- **`fine`**  
  Boolean flag to enable "fine" correlation (micro-time–resolved / higher
  resolution). If true, additional micro-time information is passed to
  `tttrlib.Correlator`.

- **`time_window`**  
  Time window (in milliseconds) used by the count‑rate filter (`CrFilterWidget`)
  when enabled. Defines the width of the sliding window for count‑rate
  evaluation.

- **`max_count_rate`**  
  Maximum allowed count rate (in kHz) within `time_window` in the count‑rate
  filter. Bursts / intervals with higher count rate are marked and can be
  suppressed.

- **`count_rate_filter`**  
  Intended master flag to enable/disable the count‑rate filter by default.
  The current GUI uses an explicit checkbox; this field reflects the desired
  default behavior.

- **`weighting`**  
  Default index for the correlation weighting mode. Mapped to entries in
  `chisurf.fluorescence.fcs.weightCalculations` and controls how statistical
  weights are computed (e.g. Suren, uniform).

---

### 1.2 `database` (embedded data handling in `chisurf.base.Data`)

Used by: `chisurf.base.Data` for embedding raw files into YAML/JSON save files.

- **`embed_data`**  
  If true, small files loaded through `Data.filename` are embedded in memory
  (and in saved YAML/JSON) instead of being referenced only by path.

- **`read_file_size_limit`**  
  Maximum file size (bytes) that will be read and possibly embedded when a
  `Data` object is created from a filename.

- **`compression_data_limit`**  
  Size threshold (bytes) above which embedded file contents are compressed with
  `zlib` before storing.

- **`embed_data_limit`**  
  Maximum size (bytes) of the *compressed* payload that will actually be
  embedded. Larger files are left unembedded and only the filename is stored.

---

### 1.3 Top‑level flags

- **`enable_experimental`**  
  Global switch that enables experimental / hidden plugins and features.
  Read in several plugin discovery paths (e.g. help browser, GUI plugin menus)
  as `chisurf.settings.cs_settings.get('enable_experimental', False)`.

- **`exceptions_on_gui`**  
  When true, installs a custom Qt exception hook (`chisurf.gui.exception_hook`)
  so that uncaught exceptions are shown in the GUI rather than silently
  printing to stderr.

- **`warn_missing_detector_setups`**  
  Controls whether a warning dialog is shown when the central
  `detector_setups.json` is missing (used by the detector wizard in
  `gui.widgets.wizard.tttr_channel_definition`). Updated via
  `set_warn_missing_detector_setups()`.

- **`fitting_message`**  
  Master flag indicating whether information dialogs should be shown around
  long‑running fits. Used by several wizards and controllers to decide if they
  display additional explanatory text.

- **`verbose`**  
  Global verbosity flag. Used in multiple modules (e.g. `chisurf.__init__`,
  `TCSPCReader.autofitrange`, photon I/O) to control the amount of diagnostic
  logging and console output.

- **`log_level`**  
  Default Python logging level for ChiSurf (10=DEBUG, 20=INFO, 30=WARNING,
  40=ERROR, 50=CRITICAL).
  Used when initializing the global logger and the GUI log dock.

- **`hidden_execute`**  
  Indicates whether certain internal operations (e.g. macro execution) are run
  without echoing the commands in the console UI. Serves as a global toggle
  for hiding implementation details from the interactive user.

- **`n_threads`**  
  Default number of CPU threads used by specific subsystems. For example,
  `structure.av.dynamic` calls `numexpr.set_num_threads(cs_settings['n_threads'])`
  to control the number of expression‑evaluation threads.

---

### 1.4 `fcs` (correlation curve weighting)

Used by: FCS utilities in `chisurf.fluorescence.fcs` and various FCS tools.

- **`weight_type`**  
  Default noise model used by `chisurf.fluorescence.fcs.noise` when computing
  statistical errors and weights for correlation curves. Common values are
  `"suren"`, `"starchev"`, or `"uniform"`.

---

### 1.5 `fortune` (message‑box extras)

Used by: `chisurf.gui.widgets.general.MyMessageBox` and
`chisurf.gui.widgets.fortune`.

- **`enabled`**  
  Master flag; if true, info dialogs may append a random “fortune cookie”
  message from the bundled fortune databases.

- **`attempts`**  
  Maximum number of attempts to pick a fortune of acceptable length.

- **`min_length`** / **`max_length`**  
  Minimum and maximum allowed length of a fortune (characters). The selection
  loop keeps drawing until a fortune within this range is found or
  `attempts` is exceeded.

---

### 1.6 `fps` (structure accessible‑volume / flexible protein simulations)

Used by: `chisurf.structure.av.*` and related structure modelling code.

- **`allowed_sphere_radius`**  
  Radius (nm) of the sphere around the attachment point that defines the
  allowed volume for linker sampling.

- **`distance_samples`**  
  Number of Monte‑Carlo samples used when generating distance distributions
  between labels.

- **`dynamic.exponential`**  
  Flag for using an exponential model in dynamic AV simulations (e.g. for
  time‑dependent quenching / motion).

- **`linknodes`**  
  Number of discretization points along the linker for geometrical AV
  calculations.

- **`simulation_grid_resolution`**  
  Base grid resolution (`dg`) used for AV density grids in
  `structure.av.static` and `structure.av.dynamic`.

- **`vdw_max`**  
  Maximum van‑der‑Waals radius (Å or nm, depending on model) used when
  constructing exclusion volumes for AV simulations.

---

### 1.7 `fret` (global FRET distance grid)

Used primarily by `chisurf.models.tcspc.fret` and helper functions.

- **`rda_min`**, **`rda_max`**  
  Minimum and maximum donor–acceptor distances (in Å) defining the global
  logarithmic `R_DA` axis used in FRET models.

- **`rda_resolution`**  
  Number of points on the logarithmic distance axis. Used to reshape
  distributions and distance grids.

- **`forster_radius`**  
  Förster radius R
a (in Å) used as the default in FRET decay and efficiency
  calculations.

- **`tau0`**  
  Donor fluorescence lifetime in the absence of FRET (ns); default for FRET
  decay simulations.

- **`kappa2`**  
  Mean orientation factor ⟨κ²⟩ assumed in FRET efficiency calculations.

- **`orientation_mode`**  
  String describing the assumed rotational dynamics model for the dipoles
  (e.g. `slow_isotropic`). Used by FRET and anisotropy models.

- **`bin_lifetime`**  
  If true, lifetime information is binned when computing FRET histograms.

- **`discriminate`**, **`discriminate_amplitude`**  
  Control optional discrimination thresholds in PDA/FRET analyses
  (e.g. remove extremely small components below `discriminate_amplitude`).

- **`lifetime_bins`**  
  Number of bins for lifetime‑resolved FRET histograms.

---

### 1.8 `gui` (graphical interface and plotting)

Sub‑sections control editor appearance, console behavior, fit windows, and
plot defaults. Many of these are read via `chisurf.settings.gui`.

#### 1.8.1 Window behavior

- **`RubberBandMove`** / **`RubberBandResize`**  
  Passed to `QMdiSubWindow.setOption` in `FitSubWindow` and other windows to
  enable live rubber‑band move/resize handles for MDI subwindows.

- **`confirm_close_fit`**  
  If true, closing a fit window prompts the user for confirmation.

- **`confirm_close_program`**  
  If true, closing the main ChiSurf window shows a confirmation dialog.

- **`fit_windows_size`**  
  Two‑element list `[width, height]` giving the default size (in pixels) of
  fit subwindows and several TTTR acquisition windows.

- **`fit_window_style`**  
  Qt stylesheet string applied to fit windows to customize their frame and
  background appearance.

- **`style_sheet`**  
  Name of the `.qss` style file used as the global GUI theme (looked up in the
  styles directory copied to the user settings folder). Updated by the
  style‑manager plugin.

- **`show_console`** / **`show_macro_edit`**  
  Control whether the embedded IPython console and macro editor panes are
  visible on startup.

- **`start_jupyter_on_startup`**  
  If true, starts an external Jupyter kernel / notebook integration when
  launching ChiSurf.

#### 1.8.2 Embedded console

- **`console_init`**  
  Multiline string executed in the embedded IPython console on startup. Used to
  configure `%matplotlib`, tab completion, imports, etc.

- **`console_style`**  
  Name of the console color theme (e.g. `linux`). Passed to the console
  widget when it is created.

- **`console_width`**, **`console_height`**  
  Initial character width and height of the embedded console widget.

#### 1.8.3 Script editor

Fields under `gui.editor` control the integrated text editor:

- **`font_family`**, **`font_size`**  
  Default font family and size for the editor.

- **`default_color`**, **`paper_color`**  
  Default text color and background color.

- **`caret_line_background_color`**  
  Highlight color for the current caret line.

- **`margins_background_color`**, **`marker_background_color`**  
  Colors used for margin gutters and marker backgrounds.

- **`language`**  
  Default language mode (e.g. `Python`) used by the editor for syntax
  highlighting.

#### 1.8.4 Fit‑models browser

- **`fit_models.n_columns`**  
  Number of columns used to arrange fit models in the model‑selection dialog.

#### 1.8.5 Plotting defaults (`gui.plot`)

Used across the plotting layer (`plots.*`, FCS/PDA widgets, PDA widgets,
protein MC plots, etc.).

- **`line_width`**  
  Default line width for curves in most plots.

- **`enable_grid`**, **`show_data_grid`**, **`show_residual_grid`**,
  **`show_acorr_grid`**  
  Flags controlling whether grid lines are shown for different plot panels
  (main data, residuals, auto‑correlation) in generic line plots.

- **`enable_region_selector`**  
  If true, enables a draggable region selector in main plots used to adjust
  fit ranges.

- **`hideTitle`**  
  If true, hides the titles of pyqtgraph `Dock` widgets in multi‑panel
  plots (e.g. AV plots, protein MC plots).

- **`label_axis`**  
  Legacy flag for axis labeling; by default axis labels are always shown for
  clarity in modern plots.

- **`pyqtgraph_config`**  
  Sub‑dictionary with low‑level pyqtgraph options such as `antialias`,
  `background`, `foreground`, or `leftButtonPan`. Applied when initializing
  pyqtgraph widgets.

- **`colors`**  
  Named colors used in most line plots:
  - **`data`** – main experimental curve
  - **`model`** – fitted model curve
  - **`residuals`** – residuals
  - **`auto_corr`** – autocorrelation curves
  - **`irf`** – instrument response function
  - **`region_selector`** / **`region_selector_alpha`** – color and opacity of
    the fit‑range selector overlay
  - **`active_transparency`**, **`inactive_transparency`** – transparencies for
    active vs. inactive curves.

---

### 1.9 `threads` (numeric backend threading)

Applied very early at startup by `env_bootstrap._apply_thread_env_from_settings`.
These values are mapped to environment variables for NumPy/Numba/MKL/OMP.

- **`numba_num_threads`**  
  Value for `NUMBA_NUM_THREADS` – maximum number of threads used by Numba’s
  parallel backend.

- **`numba_threading_layer`**  
  Value for `NUMBA_THREADING_LAYER` (e.g. `workqueue`). Controls Numba’s
  internal threading implementation.

- **`mkl_num_threads`**  
  Value for `MKL_NUM_THREADS` – maximum number of threads used by Intel MKL
  (BLAS/LAPACK).

- **`omp_num_threads`**  
  Value for `OMP_NUM_THREADS` – maximum threads for OpenMP‑using libraries
  (NumPy, SciPy, etc.).

- **`mkl_threading_layer`**  
  Value for `MKL_THREADING_LAYER` (e.g. `SEQUENTIAL`) to avoid nested
  parallelism issues.

- **`override_existing_env`**  
  If true, the above variables are applied even if already set in the OS
  environment; otherwise they only fill missing or empty variables.

---

### 1.10 `mc_settings` (protein Monte‑Carlo / AV simulations)

Used by: `chisurf.plugins.modelling.proteinmc.core.ProteinMCWorker` and related tools.

- **`append_new_structures`**  
  If true, newly generated structures are appended to existing trajectories
  instead of starting from scratch.

- **`av_filename`**  
  Default output filename for averaged structures (e.g. `out.xyz`).

- **`av_number_protein_mc`**  
  Number of protein MC samples used when computing an average structure.

- **`cluster_structures`**  
  If true, post‑process generated structures using clustering before
  summarizing.

- **`do_av_steepest_descent`**  
  Whether to run a steepest‑descent minimization on averaged structures.

- **`kt`**, **`ktAv`**  
  Effective thermal energy (kT) parameters controlling MC acceptance and
  averaging behavior.

- **`mc_mode`**  
  Name of the Monte‑Carlo mode / algorithm (e.g. `simple`).

- **`move_map`**  
  Optional move map description for which residues/atoms are allowed to move.

- **`n_iter`**  
  Total number of Monte‑Carlo iterations.

- **`n_out`**  
  Interval (in iterations) at which structures are written out.

- **`number_of_moving_aa`**  
  Number of amino acids allowed to move simultaneously in each MC step.

- **`pChi`, `pOmega`, `pPhi`, `pPsi`**  
  Probabilities for proposing moves in different torsion angles (χ, Ω, φ, ψ).

- **`pdbOut`**  
  Interval (in iterations) at which PDB snapshots are written.

- **`potentials`**  
  List of potential energy terms with a `name` and `weight` used in the
  scoring function (e.g. `H-Potential`, `Iso-UNRES`).

- **`scale`**  
  Global scaling factor for potential energies.

- **`update_rmsd`**  
  If true, RMSD values are updated continuously during MC sampling.

---

### 1.11 `optimization` (global & local fitting)

Used by: `chisurf.fitting.fit.Fit` / `FitGroup`, global‑fit machinery, MEM
regularization, and sampling routines.

- **`global_optimize_local_first`**  
  In `FitGroup.global_optimize`, controls whether individual fits are
  locally optimized before the global least‑squares run.

- **`global_threaded_model_update`**  
  If true, `GlobalFitModel.update_model` updates individual models in
  parallel threads before computing the global objective.

Sub‑sections:

- **`leastsq`** (used in local and global least‑squares)  
  Passed into `chisurf.math.optimization.leastsqbound` and SciPy‑style
  optimizers:
  - **`ftol`**, **`xtol`**, **`gtol`** – standard convergence tolerances
  - **`maxfev`** – max. number of function evaluations (0 = auto)
  - **`factor`**, **`epsfcn`** – step‑size and finite‑difference parameters
  - **`full_output`** – whether to keep full optimizer diagnostics

- **`mem`** (maximum entropy regularization; `math.optimization.mem`)  
  Controls MEM optimizer configuration:
  - **`lower_bound`**, **`upper_bound`** – bounds for the solution
  - **`factr`** – convergence factor for L‑BFGS or similar routines
  - **`maxfun`**, **`maxiter`** – iteration / function‑evaluation limits
  - **`reg_scale`** – regularization strength scaling.

- **`sampling`** (MCMC / error‑estimation sampling)  
  Used by `FittingControllerWidget.onErrorEstimate` and
  `chisurf.fitting.fit.sample_fit`:
  - **`method`** – sampling backend (e.g. `emcee`)
  - **`steps`** – number of steps per walker / chain
  - **`n_runs`** – number of independent chains or restarts
  - **`thin`** – thinning factor for recorded samples
  - **`chi2max`** – maximum χ² value allowed when accepting samples.

---

### 1.12 `parameter` (per‑parameter GUI defaults)

Used by: `chisurf.gui.widgets.fitting.widgets.parameter_settings` and
`chisurf.fitting.parameter.FittingParameter`.

- **`bounds_on`**  
  Global default for whether parameter bounds are active.

- **`decimals`**  
  Default number of decimal digits shown in parameter editors.

- **`error_color_small`**, **`error_color_large`**  
  Colors used to highlight small vs. large parameter errors in the fitting
  GUI.

- **`error_threshold_small`**, **`error_threshold_large`**  
  Thresholds that separate “small” vs. “large” relative errors.

- **`fixable`** / **`fixed`**  
  Default flags controlling whether parameters can be fixed and whether
  they start fixed.

- **`hide_bounds`**, **`hide_error`**, **`hide_fix_checkbox`**, **`hide_label`**,
  **`hide_link`**  
  Control which parameter‑editing widgets are visible by default (bounds
  editors, error display, fix checkbox, label, link controls).

---

### 1.13 `photons` (TTTR HDF5 compression)

Used in photon‑file conversion code (HDF5 via PyTables).

- **`complevel`**  
  Compression level (0–9) passed to HDF5/PyTables when writing photon
  datasets.

- **`complib`**  
  Compression library used for photon tables (e.g. `blosc`).

- **`title`**  
  Default HDF5 group name / title used when creating temporary photon
  datasets (e.g. `"spc"`).

---

### 1.14 `plugins` (plugin discovery and visibility)

Used by: plugin manager, help browser, GUI menu construction, updater plugin.

- **`disabled_models`**  
  List of model widget names that should not appear in model selection dialogs
  (e.g. experimental models).

- **`disabled_plugins`**  
  List of plugin display names that should not be loaded/shown in menus.

- **`hide_disabled_models`**  
  If true, disabled models are hidden completely from GUI lists; if false,
  they may be shown but disabled.

- **`hide_disabled_plugins`**  
  Similar to `hide_disabled_models`, for plugins.

- **`icons_enabled`**  
  Master switch for showing plugin icons in menus and toolbars where available.

- **`plugin_order`**  
  Optional mapping used to order plugins in menus. Keys are plugin names or
  categories; values are integer sort keys.

- **`toolbar_plugins`**  
  List of plugin names that should be exposed in the main toolbar for
  one‑click access (e.g. `Tools:Histogram-Microtime`, `FCS:Correlator`).

- **`updater.ignore_updates_on_startup`**  
  If true, suppresses automatic update checks at application startup.

- **`updater.check_on_startup`**  
  If true (and `ignore_updates_on_startup` is false), the updater plugin
  checks for conda or GitHub updates when ChiSurf starts.

---

### 1.15 `tcspc` (global TCSPC reader and model defaults)

Used by: `chisurf.experiments.tcspc.TCSPCReader` and various TCSPC models.

- **`dt`**  
  Default time resolution (ns per bin) for TCSPC decays when not explicitly
  provided to `TCSPCReader`.

- **`rep_rate`**  
  Default excitation repetition rate (MHz) used in decay simulations and
  some model conversions.

- **`fit_area`**  
  Fraction of the decay used for automatic fit‑range selection
  (`initial_fit_range`).

- **`fit_count_threshold`**  
  Minimum count threshold used by `initial_fit_range` when locating the
  significant part of the decay.

- **`fit_start_fraction`**  
  Fraction of the decay at which fitting typically starts (relative to the
  peak region).

- **`autoscale`**  
  If true, some TCSPC models (e.g. nuisance/background) treat the total
  photon number parameter `n0` as fixed and automatically scaled.

- **`n0`**  
  Default total photon count parameter used by TCSPC nuisance/background
  models when `autoscale` is enabled.

- **`g_factor`**  
  Global G‑factor used for anisotropy corrections in TCSPC experiments when a
  per‑dataset value is not specified.

- **`is_jordi`**  
  Flag for interpreting certain ASCII decays as Jordi‑style data.

- **`polarization`** / **`polarization_options`**  
  Default polarization mode for anisotropy decays (e.g. `vm` for vertical
  excitation / magic‑angle detection) and the list of supported codes shown in
  GUIs.

- **`rebin`**  
  Default rebinning factors `(x, y)` applied when reading TCSPC histograms
  from text/CSV.

- **`default_convolution_mode`**  
  Default convolution mode (e.g. `per` for periodic) used in TCSPC models with
  IRF convolution.

- **`convolution_on_by_default`**  
  If true, TCSPC models start with IRF convolution enabled.

- **`shift_bg_with_irf`**  
  Whether background components are shifted together with the IRF in certain
  TCSPC models (e.g. when moving the IRF peak).

- **`ts`**  
  Additional time‑shift parameter (ps or ns depending on context) used by some
  TCSPC reading or modelling routines.

---

### 1.16 `tcspc_csv` (ASCII/CSV TCSPC reader defaults)

Used by: `chisurf.experiments.tcspc.TCSPCReader` and its GUI controller via
`experiment_configs.yaml` `settings_key: tcspc_csv`.

- **`skiprows`**  
  Number of header rows to skip when reading ASCII/CSV decays.

- **`use_header`**  
  If true, header information is parsed and used to infer time axes and
  metadata where supported.

---

## 2. `experiment_configs.yaml` (experiment registry)

This file defines how high‑level "experiment types" are mapped to
reader/controller classes and model widgets. It is read through
`chisurf.experiments.load_experiment_types()` and by various GUI wizards.

### 3.1 `experiment_types`

Top‑level mapping from experiment keys to display names and visibility:

- **`<key>.name`**  
  Human‑readable name shown in the GUI (e.g. `"TCSPC"`, `"FCS"`).

- **`<key>.hidden`**  
  If true, the experiment type is not shown in generic experiment selectors
  but can still be constructed programmatically.

### 3.2 Per‑experiment sections (`tcspc`, `fcs`, `pda`, `rics`, `pch`, `structure`, `stopped_flow`, `global`)

Each experiment key has:

- **`readers`** – list of reader definitions:
  - **`reader_class`**  
    Fully‑qualified Python class path implementing the reader.
  - **`reader_params`**  
    Keyword arguments passed when constructing the reader
    (e.g. `name`, `experiment_reader`, file‑type specific hints).
  - **`controller_class`**  
    Optional Qt widget class used as GUI controller for that reader.
  - **`controller_params`**  
    Additional keyword arguments for the controller.
  - **`settings_key`**  
    Name of a sub‑section in `settings_chisurf.yaml` (e.g. `tcspc_csv`)
    providing GUI defaults for this reader.

- **`models`** – list of model widget class paths that can be used with this
  experiment type (e.g. TCSPC lifetime models, FCS models, PDA models).

These mappings allow new experiment types, readers, or models to be added or
reordered without changing Python code, by editing the YAML instead.

---

## 3. `settings_colors.yaml` (global color palette)

`settings_colors.yaml` defines a list of named colors used throughout the GUI
for plots and styling. It is loaded as `chisurf.settings.colors`.

Each entry has the form:

```yaml
- hex: "#78DBE2"
  name: Aquamarine
  rgb: "(120, 219, 226)"
```

Fields:

- **`hex`**  
  Hexadecimal RGB color code used by plotting routines (e.g. line colors in
  FCS/PDA plots, AV plots, protein MC plots).

- **`name`**  
  Human‑readable color name (primarily informational; occasionally used for
  legends or debugging).

- **`rgb`**  
  String form of the `(R, G, B)` triplet corresponding to `hex`.

The palette provides a consistent set of visually distinct colors that can be
cycled for multiple data sets or used as theme colors by different plot types.

---

## 4. `setup_defaults.json` (experiment setup persistence)

`setup_defaults.json` is stored in the user settings folder (`~/.chisurf`) and
contains persisted experiment setup parameters. This allows users to retain their
preferred settings between sessions without re-entering them on every startup.

### 4.1 File location

- **Path**: `~/.chisurf/setup_defaults.json`
- **Format**: JSON with schema version and experiment-specific defaults

### 4.2 Schema

```json
{
  "schema_version": 1,
  "last_selection": {
    "experiment_index": 0,
    "setup_index": 0
  },
  "experiments": {
    "TCSPC": {
      "TTTR Reader": {
        "module": "chisurf.experiments.tcspc",
        "class": "TCSPCTTTRReader",
        "state": {
          "reading_routine": "PTU",
          "channel_numbers": [0],
          "micro_time_coarsening": 1,
          "g_factor": 1.0
        }
      }
    },
    "RICS": {
      "RICS Reader": {
        "module": "chisurf.experiments.rics",
        "class": "RICSReader",
        "state": {
          "reading_routine": "PTU",
          "x_range": [0, 256],
          "y_range": [0, 256]
        }
      }
    }
  }
}
```

### 4.3 Behavior

- **On startup**: After creating experiment readers and controllers, ChiSurf loads
  `setup_defaults.json` (if present) and applies saved parameters to matching
  readers. The last-selected experiment and setup indices are also restored.
  Then `controller.updateUI()` is called with signals blocked to ensure the UI
  reflects the restored internal state.

- **On close**: Before the application exits, ChiSurf collects the current state
  of all readers and writes it to `setup_defaults.json`. This captures any
  parameter changes made during the session.

### 4.4 Resetting

To reset all persisted setup defaults, use the **Clear local settings** option
in the Settings menu. This deletes `setup_defaults.json` along with other local
settings.

### 4.5 Implementation notes

Each experiment controller must implement two methods for proper persistence:

- **`updateUI()`**: Reads from the reader (`chisurf.cs.current_setup`) and
  updates widgets to reflect internal state.

- **`onParametersChanged()`**: Reads from widgets and updates the reader
  attributes to persist changes.

This bidirectional sync ensures that:
1. The reader is the single source of truth for all parameters
2. UI state always matches internal state after `updateUI()`
3. Parameter changes are captured correctly for persistence
