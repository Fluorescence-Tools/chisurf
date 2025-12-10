# PDA TTTR files {#pda-tttr}

This section describes how to load Photon Distribution Analysis (PDA) data from
TTTR files (PTU/HT3/SPC) into ChiSurf using the **PDA** experiment.

## Supported file formats

The PDA TTTR reader supports TTTR files handled by `tttrlib`, in particular:

- PicoQuant PTU files (`*.ptu`)
- Becker & Hickl SPC/HT3 files (`*.spc`, `*.ht3`)

Additional formats supported by `tttrlib.TTTR` may also work, provided they
contain the required micro-time and macro-time information.

## Reader setup: PTU/HT3/SPC

In the main GUI, select:

- **Experiment**: `PDA`
- **Setup**: `PTU/HT3/SPC`

Then drop or select your TTTR files in the PDA reader widget.

The reader will:

1. Load one or more TTTR files using `tttrlib.TTTR`.
2. Apply optional burst slicing (if configured) to restrict analysis to
   selected time intervals.
3. Compute experimental S1S2 histograms via `tttrlib.Pda.compute_experimental_histograms`.
4. Attach the resulting PDA data (S1S2 matrix, probabilities, indices, etc.)
   to ChiSurf `DataCurve` objects, which can then be fitted with PDA models.

## Typical workflow

1. Configure detector channels (green/red) and micro-time windows in the PDA
   setup.
2. Load TTTR files recorded for your sample.
3. Inspect the S1S2 histogram and projections.
4. Choose an appropriate PDA model (e.g. simple or Gaussian distance model).
5. Fit the model and inspect residuals and parameter estimates.

For details on user-defined PDA models and how to extend the analysis, see the
main PDA documentation and user model guide.
