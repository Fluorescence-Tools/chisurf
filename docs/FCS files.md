# FCS file formats

This page describes the file formats supported by the **FCS** experiment in
ChiSurf and how their columns are interpreted, based strictly on the
implementation in `chisurf.experiments.fcs` and
`chisurf.fio.fluorescence.fcs.*`.

Configured FCS readers (see `experiment_configs.yaml`):

- **Seidel Kristine** (`experiment_reader: kristine`)
- **FCS-CSV** (`experiment_reader: csv`)
- **Zeiss Confocor3** (`experiment_reader: confocor3`)
- **China FCS (MATLAB)** (`experiment_reader: china-mat`)
- **PyCorrFit CSV** (`experiment_reader: pycorrfit`)
- **ALV-Correlator .asc** (`experiment_reader: alv`)

All of these are normalized internally into an `FCSDataset` structure with
fields like `correlation_times`, `correlation_amplitudes`,
`correlation_amplitude_weights`, `intensity_trace` and `meta_data`, but the
on-disk column conventions differ per format.

---

## Seidel Kristine `.cor` {#fcs-kristine}

Implemented in `chisurf.fio.fluorescence.fcs.kristine` as
`read_kristine` / `write_kristine`.

### Column layout (write side)

`write_kristine` writes a text file with either **3** or **4** numeric columns
per row:

1. **Correlation time** (`correlation_time`)
2. **Correlation amplitude** (`correlation_amplitude`)
3. **Acquisition / count-rate column** (`col_3`)
   - Mostly zeros.
   - Special rows:
     - Row 0: `acquisition_time` (s)
     - Row 1: `mean_countrate` (kHz)
4. **Optional correlation-amplitude uncertainty**
   - Present only if an uncertainty array is provided.

### Column usage when reading

`read_kristine` does:

```python
data = np.loadtxt(filename).T
x, y = data[0], data[1]
# keep only x > 0
idx = np.where(x > 0.0)
x = x[idx]
y = y[idx]
dur, cr = data[2, 0], data[2, 1]
try:
    w = 1.0 / data[3][idx]   # experimental errors if 4th column exists
except IndexError:
    w = 1.0 / chisurf.fluorescence.fcs.noise(x, y, dur, cr, weight_type='suren')
```

So the reader interprets the columns as:

- **Column 0** (`data[0]`):
  - Correlation lag time `τ` (as stored by the correlator; used directly as
    `correlation_times`).
  - Only strictly positive entries are kept (`x > 0`).
- **Column 1** (`data[1]`):
  - Experimental correlation `G(τ)` (no offset is added or subtracted).
- **Column 2** (`data[2]`):
  - Row 0: `acquisition_time` in seconds.
  - Row 1: `mean_count_rate` in kHz.
  - Other rows: not used by the reader (typically zeros).
- **Optional column 3** (`data[3]`):
  - Per-point uncertainty `σ_G(τ)` for the correlation amplitude. The reader
    uses `weight = 1 / σ` as `correlation_amplitude_weights`.

If the 4th column is missing, theoretical noise-based weights are computed
from `x`, `y`, `acquisition_time` and `mean_count_rate`.

---

## FCS-CSV (generic ASCII) {#fcs-csv}

When `experiment_reader: csv`, `read_fcs` uses the generic `Csv` loader in
`chisurf.fio.ascii` and then interprets the first three data rows as:

```python
csv = chisurf.fio.ascii.Csv()
csv.load(filename=filename, ...)
x, y = csv.data[0], csv.data[1]
ey = csv.data[2]
```

### Expected columns

The CSV file must contain at least **three numeric columns** after any header:

1. **Correlation time**
   - Interpreted as lag time `τ` for the FCS curve.
2. **Correlation amplitude**
   - Experimental correlation `G(τ)`.
3. **Uncertainty / error**
   - Per-point uncertainty `σ_G(τ)` for the correlation amplitude.

Additional columns may be present but are ignored by this reader. Header
handling, delimiter and decimal-comma support are provided by the
`chisurf.fio.ascii.Csv` class.

---

## Zeiss Confocor3 `.fcs` {#fcs-confocor3}

Implemented in `chisurf.fio.fluorescence.fcs.confocor3` as
`read_zeiss_fcs` on top of `openFCS` / `openFCS_Multiple` / `openFCS_Single`.

The `.fcs` files are text-based and contain different labeled sections.
The reader primarily uses two of them:

- A **correlation section** (`"Correlation"` or `"Correlation (Multi, Averaged)"`)
- A **count-rate section** (`"Count Rate"`)

### Correlation section

For multi-curve files (`openFCS_Multiple`), the correlation part is parsed as:

```python
# correlation rows from the FcsDataSet CorrelationArray
corr.append((float(row[3]) * 1000, float(row[4]) - 1))  # tau [ms], corr - 1
...
correlation_time = correlation[:, 0]          # ms
correlation_amplitude = correlation[:, 1] + 1.0
```

Thus each correlation row in the file has at least:

- **Column 3**: lag time `τ` in seconds → converted to **milliseconds**.
- **Column 4**: `G(τ) - 1`
  - The reader adds `+1.0` to get `G(τ)` as stored in
    `correlation_amplitudes`.

For single-curve files (`openFCS_Single`), correlation rows look like:

```python
# tau in ms, corr-function
corr.append((float(row[0]), float(row[1]) - 1))
```

So the **first column** is lag time in **ms** and the **second column** is
`G(τ) - 1`.

### Intensity trace section

In the multiple-curve variant, the **Count Rate** section is read as:

```python
# tau in ms, trace in kHz
trace.append((float(row[3]) * 1000, float(row[4]) / 1000))
```

So each row in the count-rate block contains at least:

- **Column 3**: time in seconds → converted to **ms**.
- **Column 4**: count rate in Hz → converted to **kHz**.

In the single-curve variant:

```python
# tau in ms, trace in kHz
trace.append((float(row[0]) * 1000, float(row[1]))
```

The reader uses the trace to compute acquisition time and mean count rate,
which in turn are used to derive weights for `correlation_amplitude_weights`.

---

## China FCS MATLAB `.mat` {#fcs-china-mat}

Implemented in `chisurf.fio.fluorescence.fcs.china` as `read_china_mat`.

The MATLAB file is expected to contain arrays with specific names, for example:

- `AA`, `BB`, `AAxB`: correlation curves
- `IntA`, `IntB`: intensity traces

The reader uses a `correlation_keys` mapping:

```python
correlation_keys = {
    'Auto_AA':  {'Intensity': 'IntA',       'Correlation': 'AA'},
    'Auto_BB':  {'Intensity': 'IntB',       'Correlation': 'BB'},
    'Cross_Axb':{'Intensity': ['IntA','IntB'],'Correlation': 'AAxB'},
}
```

### Correlation arrays

For each correlation key, `read_china_mat` interprets the matrix columns as:

```python
correlation_time = m[correlation_key][:, 0]
correlation_amplitude = m[correlation_key][:, measurement_number] + 1.0
```

So in each correlation matrix (e.g. `AA`):

- **Column 0**: lag time `τ` (as stored in the `.mat` file).
- **Columns 1..N-1**: correlation curves for individual measurements, stored
  as `G(τ) - 1`. The reader **adds 1.0** to obtain `G(τ)`.

### Intensity arrays

For each measurement:

```python
# single-channel case
intensity_time = m[intensity_key][:, 0]
intensity       = m[intensity_key][:, measurement_number]

# dual-channel case (Cross_Axb)
intensity = 0
for k in intensity_key:   # 'IntA', 'IntB'
    intensity += m[k][:, measurement_number]
intensity_time = m[k][:, 0]
```

So in each intensity matrix (`IntA`, `IntB`):

- **Column 0**: time points (units as stored in the `.mat` file).
- **Columns 1..N-1**: count-rate traces for individual measurements.

The reader derives:

- `acquisition_time`: last time point of the intensity trace.
- `mean_count_rate`: `sum(intensity) / (acquisition_time * 1000.0)`.

These are used with the correlation data to compute
`correlation_amplitude_weights` via `chisurf.fluorescence.fcs.noise`.

---

## PyCorrFit CSV {#fcs-pycorrfit}

Implemented in `chisurf.fio.fluorescence.fcs.pycorrfit` as `read_pycorrfit`
(on top of `openCSV`). The format is flexible but the reader assumes a
PyCorrFit-style structure:

- Optional header lines starting with `#`.
- A **correlation section**.
- One or two **trace sections** marked by `# BEGIN TRACE` and
  `# BEGIN SECOND TRACE`.

### Correlation section columns

From the `openCSV` docstring and parsing logic:

- Minimal form (two columns):

  ```text
  tau[s]   G(τ)
  ```

  - Column 0: lag time `τ` in seconds (later converted to ms internally).
  - Column 1: correlation amplitude `G(τ)`.

- Extended form (five columns):

  ```text
  # Channel (tau [s])    Experimental correlation    Fitted correlation    Residuals    Weights [model function]
  2.0e-07    1.56e-01    1.54e-01    2.69e-03    7.31e-03
  ...
  ```

  - Column 0: lag time `τ` in seconds.
  - Column 1: experimental correlation `G(τ)`.
  - Column 2: fitted correlation (ignored by ChiSurf’s reader).
  - Column 3: residuals (ignored).
  - Column 4: weights from the model function; used as pointwise weights.

Internally, the reader stores the correlation as an array of `(τ_ms, G(τ))`
with time converted to milliseconds.

### Trace sections

- After `# BEGIN TRACE`:

  ```text
  time[s]   intensity[kHz]
  ```

  - Column 0: time `t` in seconds.
  - Column 1: intensity (kHz).

- After `# BEGIN SECOND TRACE` (for cross-correlation): same two-column
  layout for the second channel.

The reader collects:

- `Duration`: measurement duration in seconds (from a `#   duration [s]` line).
- `Count rates`: a list of average count rates parsed from lines containing
  `"avg. signal"`.

These are combined to compute noise-based
`correlation_amplitude_weights` via `chisurf.fluorescence.fcs.noise`.

---

## ALV-Correlator `.asc` {#fcs-alv}

Implemented in `chisurf.fio.fluorescence.fcs.asc_alv` as `read_asc`, built on
`openASC` / `openASC_old` / `openASC_ALV_7004`.

Two related text formats are supported:

- Classic **ALV-6000** `.asc` files (`openASC_old`).
- Newer **ALV-7004/USB** `.asc` files (`openASC_ALV_7004`).

Both provide a **Correlation** section and a **Count Rate** section.

### Classic ALV-6000 `.asc` (openASC_old)

From the docstring and parsing logic:

- **Correlation** section (quoted header `"Correlation"`):

  ```text
  tau[ms]   G(τ)
  ```

  - Column 0: lag time `τ` in milliseconds.
  - Column 1: correlation amplitude `G(τ)`.

- **Count Rate** section (quoted header `"Count Rate"`):

  ```text
  time[s]   count_rate[kHz]
  ```

  - Column 0: time in seconds.
  - Column 1: count rate in kHz.

The reader bins and splits traces as needed (for multiple curves or dual
channels) and constructs:

- `correlation_times`: from the first column of the correlation section.
- `correlation_amplitudes`: from the second column, often converted to
  `G(τ)` by adding 1.0.
- `intensity_trace_times`: from the first column of the Count Rate section
  (converted to seconds or ms depending on mode).
- `intensity_trace`: count-rate traces assembled from one or two channels.

### ALV-7004/USB `.asc` (openASC_ALV_7004)

In the newer format, the correlation and count-rate blocks look like:

```text
"Correlation"
  tau   C1   C2   C3   C4
  ...

"Count Rate"
  time[s]   CR1   CR2   CR3   CR4
  ...
```

- **Correlation** rows (parsed into `allcorr`):
  - Column 0: lag time `τ`.
  - Columns 1..4: correlation curves for up to four channels or
    auto/cross-correlations.

- **Count Rate** rows (parsed into `alltrac`):
  - Column 0: time in seconds (later converted to ms internally).
  - Columns 1..4: count rates for up to four channels.

`read_asc` then builds one or more `FCSDataset` entries by:

- Selecting the appropriate correlation column (AC1, AC2, CC12, CC21, etc.).
- Constructing one or two intensity traces from the corresponding count-rate
  columns.
- Converting intensity time to seconds and deriving `acquisition_time` and
  `mean_count_rate`.
- Computing `correlation_amplitude_weights` via
  `chisurf.fluorescence.fcs.noise`.

---

## Notes on additional FCS backends

The low-level `read_fcs` dispatcher also supports:

- PicoQuant **`.dat`** (`reader_name: 'pq.dat'` → `pq_dat.read_dat`)
  - Binary/ASCII format with triplets of columns for each correlation:
    - Col 0: `τ`.
    - Col 1: `G(τ)`.
    - Col 2: error `σ_G(τ)` (0 values replaced by large errors internally).
- YAML FCS (`reader_name: 'yaml'` → `fcs_yaml.read_yaml`, currently stubbed).
- PicoQuant **`.pqres`** (`reader_name: 'pqres'` → `chisurf.fio.fluorescence.pqres.read_pqres_fcs`).

These are not exposed as separate FCS readers in the default GUI, but follow
similar conventions: correlation times and amplitudes in the first two columns
or fields, optional errors/weights in a third.
