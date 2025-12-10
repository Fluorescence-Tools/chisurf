# TCSPC file readers

## TXT/CSV {#tcspc-txtcsv}

### Purpose

The **TCSPC TXT/CSV** reader loads time‑correlated single photon counting (TCSPC) decays
from plain text or CSV files and converts them into ChiSurf TCSPC datasets that can be
used for lifetime fitting and FRET analysis.

You typically use this reader when you have:

- Exported decays from vendor software as text / CSV tables.
- TCSPC matrices (multiple decays in columns or rows) in a single file.
- Jordi fast‑rotating dye measurements for estimating the anisotropy G‑factor.

When you select the **TCSPC** experiment and the **TXT/CSV** file type, the
"File parameters" group shows two stacked panels:

- A **CSV input** panel (generic ASCII/CSV reader).
- A **TCSPC TXT/CSV** panel (TCSPC‑specific parameters such as `dt`, polarization, G‑factor).

Clicking the question‑mark button next to the experiment selection opens this help
view with the filter pre‑filled for this reader.

---

### Supported file formats

The TCSPC TXT/CSV reader ultimately calls `chisurf.fio.fluorescence.tcspc.read_tcspc_csv`,
which accepts data in the following forms:

- **Two‑column decay**
  - First column: time axis in channel units (or arbitrary units that you convert via `dt`).
  - Second column: counts in each time bin.
  - Optionally additional columns are ignored or selected via the CSV panel.

- **Matrix of decays**
  - One column contains the time axis.
  - One or more additional columns contain decay curves (e.g. different channels, ROIs, or repeats).
  - The `Matrix columns` setting in the TCSPC panel tells the reader which columns to interpret
    as TCSPC channels.

- **Jordi format (fast‑rotating dye)**
  - Special case used for anisotropy calibration.
  - Select **Jordi mode** in the TCSPC panel (`is_jordi`), then load the Jordi file.
  - The reader uses `chisurf.fio.jordi.read_jordi` and can derive `g_factor` and VV/VH/VM
    channels directly from the file.

Delimiters (comma, semicolon, tab, space) are usually detected automatically when
**Use header** is enabled in the CSV panel and `is_jordi` is *not* checked.

---

### CSV input panel (generic ASCII/CSV settings)

The **CSV input** panel comes from `chisurf.gui.widgets.fio.CsvWidget`. Its
settings control how the ASCII/CSV file is parsed before it is interpreted as
TCSPC data:

- **Use header**
  - If checked, the first non‑skipped line is treated as a header line.
  - Column indices can then be resolved from header names where supported.

- **Skip rows** (`skiprows`)
  - Number of initial lines to ignore before reading numeric data.
  - Use this to skip instrument headers, comments, or metadata blocks.

- **Columns / colspecs**
  - Free‑form column specification string, stored as `cs.current_setup.colspecs`.
  - Use this to pick the X/Y/error columns when your file contains many columns.

- **CSV type / reading routine**
  - **Auto**: let ChiSurf infer the format based on the file extension.
  - **CSV**: force CSV/ASCII parsing.
  - **FWF**: fixed‑width formatted files.
  - **YAML**: interpret the file as a YAML data description.

- **Error columns**
  - You can optionally specify columns for `error_x` and `error_y` (standard
    deviations for X and Y).
  - These are mapped to `cs.current_setup.col_ex`, `col_ey`, `col_x`, `col_y`.

All changes in this panel are propagated into `cs.current_setup` via the
Python console so that the TCSPC reader sees a consistent configuration.

---

### TCSPC TXT/CSV panel (TCSPC‑specific settings)

The lower panel, provided by `CsvTCSPCWidget`, configures how the ASCII data
is interpreted as a TCSPC experiment:

- **Time resolution `dt`**
  - Effective width of one time bin in **nanoseconds**.
  - If **Rebin in Y** is active and "keep dt" is checked, the widget rescales
    `dt` so that the physical time axis remains consistent after rebinning.

- **Repetition rate `rep_rate`**
  - Excitation repetition rate in **MHz**.
  - Used by TCSPC models for parameters like total measurement time and
    convolution.

- **Rebin (X/Y)**
  - `rebin_y`: number of consecutive time bins summed into one new bin.
    Larger values reduce noise at the cost of temporal resolution.
  - `rebin_x`: number of decays aggregated along the X‑axis (for matrices with
    multiple rows or segments).

- **Polarization mode (`polarization`)**
  - `vm`: magic‑angle or total intensity.
  - `vv`: parallel polarization channel only.
  - `vh`: perpendicular polarization channel only.
  - `vv/vh`: two‑channel mode providing both VV and VH decays.

- **G‑factor (`g_factor`)**
  - Anisotropy correction factor used when combining VV and VH channels into VM
    or when computing anisotropy curves.
  - Often obtained from a Jordi fast‑rotating dye measurement.

- **Matrix columns (`matrix_columns`)**
  - Which columns of the numeric matrix correspond to the TCSPC channels.
  - For non‑Jordi files, the CSV loader reads `usecols=matrix_columns`.
  - Typical examples:
    - `0 1` → time in column 0, intensity in column 1.
    - `0 1 2` → time + two intensity channels, which will be turned into
      multiple decays.

- **Jordi mode (`is_jordi`) and header usage (`use_header`)**
  - When **Jordi mode** is enabled:
    - The reader switches to `chisurf.fio.jordi.read_jordi` and ignores
      standard `matrix_columns`.
    - Available polarization channels (VV, VH, VM) are taken from the Jordi
      file, and `g_factor` can be read from metadata.
  - When Jordi is **off**, the reader expects a generic ASCII/CSV file and
    uses `matrix_columns`, `use_header`, and the CSV panel configuration.

- **VH shift (`vh_shift`)** (if present)
  - Integer shift (in bins) applied to the VH channel before combining VV and
    VH (Jordi mode).
  - Positive values shift VH to later times; negative values to earlier times.
  - Useful to compensate for small timing offsets between detection channels.

All changes you make in this panel are written back into `cs.current_setup`
when parameters change, so subsequent data loads and fits use the same setup.

---

### Jordi files {#tcspc-jordi}

Jordi files are a special TCSPC format used primarily for **anisotropy calibration**.
They typically contain a single column of intensity values which is interpreted as:

- First half of the values → **VV** (parallel) channel
- Second half of the values → **VH** (perpendicular) channel

In the **TCSPC TXT/CSV** reader, Jordi files are handled by enabling **Jordi mode**
in the TCSPC panel (`is_jordi`). When this mode is active:

- The reader uses `chisurf.fio.jordi.read_jordi` under the hood.
- Available polarization channels (VV, VH, VM) are taken directly from the Jordi file.
- `g_factor` and related anisotropy parameters can be read from the file metadata
  when present.

For detailed inspection and refinement of the **g‑factor** and **VH shift**, you can
use the dedicated **Jordi G‑Factor Calculator** plugin, which provides:

- Interactive tail-matching of VV and VH decays.
- Background correction and visualization of corrected vs. uncorrected decays.
- Convenient copyable fields for the resulting `g_factor` and its uncertainty.

The typical workflow is:

1. Acquire or load a Jordi calibration file.
2. Use the **Jordi G‑Factor Calculator** plugin to determine a robust `g_factor`
   (and optionally a VH shift).
3. Enter the resulting parameters into the **TCSPC TXT/CSV** panel and enable
   **Jordi mode** for the corresponding datasets.

---

### Typical workflow

1. **Select experiment and reader**
   - In the *Read data* dock, choose **Experiment → TCSPC**.
   - In **File type**, choose **TXT/CSV**.

2. **Configure CSV parsing**
   - Set **Skip rows** so that the first data row corresponds to the start of the time axis.
   - Enable **Use header** if your file has a column header line.
   - Adjust column indices (X, Y, errors) if needed.

3. **Configure TCSPC parameters**
   - Set `dt` to the correct time‑per‑channel in ns.
   - Adjust **Rebin** if you want to reduce noise.
   - Choose **Polarization** mode depending on available channels.
   - Set or confirm the **G‑factor** if you do anisotropy analysis.

4. **(Optional) Jordi calibration**
   - Enable **Jordi mode** and load a Jordi VV/VH file.
   - Use the Jordi G‑factor inspector button to refine `g_factor` and `vh_shift`.

5. **Load data**
   - Click the **Data** button (or drop files onto the *Drop files here* area).
   - The resulting decays appear in the dataset list and can be used for
     lifetime fitting.

This document is shown automatically when you click the question‑mark button
next to the experiment / file‑type selection while the **TCSPC TXT/CSV**
reader is active.

---

## TTTR-file {#tcspc-tttr}

### Purpose

The **TTTR-file** reader converts time-tagged, time-resolved (TTTR) photon
streams into TCSPC decay histograms using `tttrlib`. It is implemented by
`chisurf.experiments.tcspc.TCSPCTTTRReader` and the
`TCSPCTTTRReaderControlWidget` controller.

Use this reader when you have photon-stream data from hardware that stores
individual photon arrival times (e.g. PicoQuant PTU/HT3 or Becker & Hickl SPC
TTTR formats) and want to build decays by integrating the micro-time axis.

When you select **Experiment → TCSPC** and **File type → TTTR-file**, the
"File parameters" group shows a compact TTTR panel:

- **Routine** (`reading_routine`)
  - Drop-down list of TTTR container types, e.g. `PTU`, `HT3`, `SPC132`,
    `SPC630`.
  - Passed as the second argument to `tttrlib.TTTR(filename, reading_routine)`.

- **Channels** (`channel_numbers`)
  - Comma-separated list of routing channel indices (e.g. `0, 3`).
  - The reader calls `tttr.get_tttr_by_channel(self.channel_numbers)` to
    restrict the TTTR stream to these channels before histogramming.

- **Binning** (`micro_time_coarsening`)
  - Discrete micro-time binning factor (`1`, `2`, `4`, `8`, `16`).
  - Used as the argument to
    `tttr_selected.get_microtime_histogram(self.micro_time_coarsening)`.

### Data generation

For each selected TTTR file, the reader:

1. Constructs a `tttrlib.TTTR` instance with the chosen routine.
2. Selects photons from the specified channels.
3. Computes a micro-time histogram with the configured coarsening factor.
4. Converts the micro-time axis from seconds to nanoseconds (`x *= 1e9`).
5. Trims trailing zero bins so the decay stops at the last non-zero count.
6. Creates a `chisurf.data.DataCurve` with:
   - `x`: micro-time in ns,
   - `y`: counts per bin,
   - `ey`: Poisson noise via `chisurf.fluorescence.tcspc.counting_noise(y)`,
   - `experiment`: the current TCSPC experiment,
   - `data_reader`: the `TCSPCTTTRReader` instance.

The curve name is derived from the filename and channel list, e.g.
`"file_ch(0,3)"`.

### Typical workflow

1. **Select experiment and reader**
   - In the *Read data* dock, choose **Experiment → TCSPC**.
   - In **File type**, choose **TTTR-file**.

2. **Configure TTTR parameters**
   - Set **Routine** to match the acquisition container (e.g. `PTU`, `HT3`).
   - Enter the routing **Channels** to include in the decay.
   - Choose a **Binning** factor for micro-time (trade-off between resolution
     and noise).

3. **Load data**
   - Click **Data** and select one or more TTTR files.
   - The reader builds decay histograms and adds them as datasets.

---

## Becker-SDT {#tcspc-sdt}

### Purpose

The **Becker-SDT** reader loads Becker & Hickl `.sdt` histogram files using
`chisurf.fio.fluorescence.sdtfile.SdtFile`. It is implemented by
`TCSPCSetupSDTWidget`, which wraps a `TcspcSDTWidget` for inspecting and
selecting curves.

Use this reader when you have TCSPC decays stored as Becker & Hickl SDT
histograms rather than raw TTTR streams or ASCII/CSV files.

### SDT widget behavior

`TcspcSDTWidget` performs the low-level SDT handling:

- **File** (`filename`)
  - The selected `.sdt` file, opened via `SdtFile(filename)`.
  - The widget populates a **Curve** combo box with indices `0..n_curves-1`.

- **Curve selection** (`curve_number`)
  - Chooses which internal SDT histogram to expose.
  - `ph_counts` returns the selected histogram as a NumPy array.

- **Time axis** (`times`)
  - Uses `self._sdt.times[0] * 1e9` to convert the hardware time base into
    **nanoseconds**.

- **Repetition rate** (`rep_rate`)
  - Computed from SDT `measure_info[curve]['rep_t']` as
    `1.0 / (rep_t * 1e-3)` → **MHz**.

- **Curve construction**
  - `curve` builds a `chisurf.data.DataCurve` with:
    - `x`: time in ns,
    - `y`: photon counts,
    - `ey`: Poisson noise via `chisurf.fluorescence.tcspc.counting_noise`,
    - `name`: `"<filename> _ <curve_number>"`.

`TCSPCSetupSDTWidget.read` iterates over all available curves in the file and
returns a list of `DataCurve` objects, making each curve available as a
separate dataset.

### Typical workflow

1. **Select experiment and reader**
   - In the *Read data* dock, choose **Experiment → TCSPC**.
   - In **File type**, choose **Becker-SDT**.

2. **Open SDT file**
   - Use the `...` button to select a `.sdt` file.
   - Inspect header/metadata in the text area if needed.

3. **Select curves**
   - Use the **Curve** combo box to preview individual histograms.
   - The reader will convert all curves into separate datasets when reading.

4. **Load data**
   - Click **Data** in the main GUI to import the SDT curves as TCSPC
     datasets.

---

## Simulator {#tcspc-simulator}

### Purpose

The **TCSPC Simulator** reader generates synthetic TCSPC decays based on a
user-defined lifetime spectrum. It is implemented by
`chisurf.experiments.tcspc.TCSPCSimulatorSetup` and configured via the
`TCSPCSimulatorSetupWidget`.

Use this when you want to test models, fitting routines, or instrument
response handling without loading real data.

### Simulator parameters

The simulator is configured via the **TCSPC simulator** panel
(`TCSPCSimulatorSetupWidget`, see `tcspc_simulator.ui`). It exposes the
following core parameters:

- **Name** (`sample_name`)
  - Logical name for the simulated dataset.

- **Lifetime spectrum** (`lifetime_spectrum`)
  - Comma-separated list of numeric values passed to
    `chisurf.fluorescence.general.calculate_fluorescence_decay`.
  - Interpreted as component lifetimes / amplitudes according to that
    function.

- **n TAC** (`n_tac`)
  - Number of time channels in the simulated decay (default `4096`).

- **Peak count** (`p0`)
  - Peak photon count scale (stored on the setup; the exact interpretation is
    delegated to the decay generation routine).

- **dt [ns]** (`dt`)
  - Time-per-channel in nanoseconds (default `0.0141`).

- **Instrument response function** (`instrument_response_function`)
  - Optional IRF widget/curve selected in the **Instrument response
    function** group box.
  - The GUI offers two options:
    - A **Select IRF** button that picks a dataset from the imported
      TCSPC curves.
    - A Gaussian fallback defined by *Gaussian IRF mean [ns]* and
      *sigma [ns]* when no dataset is selected.

Internally, `TCSPCSimulatorSetup.read` and the simulation preview
(`TCSPCSimulatorSetupWidget._simulate_decay`) perform the following steps:

1. Builds a time axis `x = arange(n_tac) * dt`.
2. Calls `calculate_fluorescence_decay(lifetime_spectrum=self.lifetime_spectrum,
   time_axis=x)` to obtain a model decay `y`.
3. Computes Poisson noise via `chisurf.fluorescence.tcspc.counting_noise(y)`.
4. Wraps the result in a `DataCurve` and returns a `DataCurveGroup`
   containing the single simulated dataset.

In the GUI, the simulator panel also offers a **Simulation preview**
section with:

- A **Simulate** button that runs the same decay computation (including
  optional IRF convolution and Poisson noise) and overlays the IRF on a
  log‑scaled preview plot.
- An **Add** button that takes the latest simulated trace and appends it
  to `chisurf.imported_datasets` as an `ExperimentDataCurveGroup`, so it
  behaves like any other TCSPC dataset in selectors and fits.
- A small `...` tool button next to the *lifetime spectrum* text field
  that can load a spectrum from a CSV/text file.

#### Lifetime spectrum examples

The lifetime spectrum is stored as an interleaved list of amplitudes and
lifetimes. For example, the default

```text
0.4, 1.1, 0.6, 4.1
```

corresponds to two components

- component 1: amplitude `0.4`, lifetime `1.1 ns`
- component 2: amplitude `0.6`, lifetime `4.1 ns`

You can either type this string directly into the *lifetime spectrum*
field or use the `...` loader button. The loader accepts:

- A **1‑column file** containing the interleaved values, e.g.

  ```text
  0.4
  1.1
  0.6
  4.1
  ```

- A **2‑column file** with amplitudes and lifetimes in separate
  columns, e.g.

  ```text
  # amp   tau [ns]
  0.4     1.1
  0.6     4.1
  ```

In the 2‑column case the widget automatically converts the table into
the interleaved representation and updates the text field. Any change to
the text field immediately updates `cs.current_setup.lifetime_spectrum`.

### Typical workflow

1. **Select experiment and reader**
   - In the *Read data* dock, choose **Experiment → TCSPC**.
   - In **File type**, choose **Simulator**.

2. **Configure decay parameters**
   - Set **Name**, **lifetime spectrum**, **n TAC**, **Peak count**, and
     **dt [ns]**.
   - Optionally select or configure an **Instrument response
     function**.

3. **Preview the simulated decay**
   - Click **Simulate** to compute and display the decay (yellow) and IRF
     (red) on the preview plot.

4. **Generate and store data**
   - Either click **Data** (reader‑style) or click **Add** in the
     Simulation preview. In both cases a synthetic TCSPC decay is created
     and added as a dataset, ready for fitting like any other TCSPC
     curve.
