# PCH TTTR files {#pch-tttr}

This section describes how to load Photon Counting Histogram (PCH) data from
TTTR files into ChiSurf using the **PCH** experiment.

## Supported input formats

The PCH reader uses `tttrlib.TTTR` and currently focuses on time-tagged
TTTR formats such as PicoQuant PTU files. Other TTTR formats supported by
`tttrlib` may also work, provided they contain the required photon arrival
information.

## Reader setup: PCH (TTTR)

In the main GUI, select:

- **Experiment**: `PCH`
- **Setup**: `PCH (TTTR)`

Then select or drop a TTTR file (e.g. `*.ptu`) in the PCH reader widget.

The reader will:

1. Load the TTTR data using `tttrlib.TTTR` with the configured `reading_routine`.
2. Select the specified routing channel and micro-time range.
3. Build a photon counting histogram over time bins defined by `bin_time_us`.
4. Construct a ChiSurf `DataCurve` where the x-axis indexes photon counts
   and the y-axis contains the observed PCH, with uncertainties derived from
   counting statistics.

## Typical workflow

1. Configure basic PCH settings (detector channel, bin time, micro-time range)
   in the PCH controller.
2. Load a TTTR file recorded under stationary conditions.
3. Inspect the PCH curve to verify that the dynamic range and shape are
   reasonable (no obvious saturation or truncation).
4. Choose an appropriate PCH model from the model list (e.g. multi-component
   PCH model).
5. Fit the model and inspect residuals and extracted parameters such as
   molecular brightness and number of molecules.

For more advanced PCH analysis and model details, refer to the main PCH
documentation and associated ChiSurf manuals.
