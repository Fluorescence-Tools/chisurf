# PRD-07: Plugin MFDB Integration

## Goal

High-priority plugins register their results in MFDB via the result registry (PRD-03).
This is not a rewrite -- just add one `register_result()` call at the output point of
each plugin.

## Principle

Every plugin integration follows the same pattern:

1. Find where the plugin produces its output (file, array, plot data)
2. Add `register_result()` after that point
3. Wrap in try/except so the plugin works without MFDB
4. Add `sample_id` propagation if the plugin has a GUI

That's it. No other changes to the plugin.

## Priority Tiers

### Tier 1: smFRET Workflow (do first)

These plugins are on the critical path for smFRET data processing.

### Tier 2: Supporting Analysis

These plugins produce results that should be archived but are not on the critical path.

### Tier 3: Future

These plugins would benefit from MFDB but are low priority.

---

## Tier 1 Plugins

### 1. burst_selection (already covered in PRD-04)

Skip -- see PRD-04.

### 2. lltf (Lazy Lifetime Fitting)

**Plugin path**: `chisurf/plugins/fluorescence_decay/lltf/`

**What it produces**: Lifetime fit results (amplitudes, lifetimes, chi-squared).

**Where to add**: Find the function that runs the fit and returns results. Look for
where chi-squared is computed or where results are displayed.

**What to add**:

```python
try:
    from chisurf.core.mfdb.result_registry import register_fit_result

    register_fit_result(
        fit_data={
            "chi2_reduced": chi2_r,
            "lifetimes": lifetime_values,
            "amplitudes": amplitude_values,
            "model_name": model.name,
        },
        parent_artifact_id=dataset_artifact_id,
        sample_id=sample_id,
        parameters={
            "chi2_reduced": chi2_r,
            "tau1": {"value": tau1, "error": tau1_err},
            "a1": {"value": a1, "error": a1_err},
            # ... for each component
        },
    )
except Exception:
    pass
```

**How to get `dataset_artifact_id`**: The dataset should have an `artifact_id` in its
metadata (set during import, see PRD-02 Task 5). Access via
`fit.data.meta_data.get("artifact_id", "")`.

**How to get `sample_id`**: From the dataset metadata:
`fit.data.meta_data.get("sample_id", "")`.

### 3. maxent_decay (Maximum Entropy)

**Plugin path**: `chisurf/plugins/fluorescence_decay/maxent_decay/`

**What it produces**: Lifetime distributions, FRET distance distributions.

**Same pattern as lltf**: Find where the MEM result is computed. Register:

```python
register_result(
    kind="fit_result",
    data={
        "lifetime_distribution": distribution_array.tolist(),
        "method": "maximum_entropy",
    },
    parent_artifact_id=dataset_artifact_id,
    sample_id=sample_id,
    operation_type="local_fit",
    parameters={"chi2": chi2_value},
)
```

### 4. tr_anisotropy (Time-Resolved Anisotropy)

**Plugin path**: `chisurf/plugins/fluorescence_decay/tr_anisotropy/`

**What it produces**: Rotation correlation times, anisotropy parameters, g-factor.

**Register**: After the anisotropy fit completes:

```python
register_result(
    kind="fit_result",
    data={
        "rotation_times": rho_values,
        "g_factor": g_factor_value,
        "r0": r0_value,
        "r_inf": r_inf_value,
    },
    parent_artifact_id=dataset_artifact_id,
    sample_id=sample_id,
    operation_type="local_fit",
    parameters={
        "rho1": {"value": rho1, "error": rho1_err},
        "g_factor": g_factor_value,
        "r0": r0_value,
    },
)
```

**Also**: If g-factor is determined here, register it as a calibration (PRD-05).

### 5. ndxplorer (MFD Analysis)

**Plugin path**: `chisurf/plugins/ndxplorer/`

**What it produces**: Multi-parameter histograms, FRET efficiency distributions,
population selections.

**Register**: After histogram computation:

```python
register_result(
    kind="processed_data",
    data={
        "histogram_type": "2D_ES",
        "x_bins": x_bins.tolist(),
        "y_bins": y_bins.tolist(),
        "counts": histogram.tolist(),
    },
    parent_artifact_id=burst_artifact_id,
    sample_id=sample_id,
    operation_type="population_selection",
    metadata={"parameters": ["E", "S"]},
)
```

### 6. microtime_histogram (TTTR Microtime Histogram)

**Plugin path**: `chisurf/plugins/tttr/microtime_histogram/`

**What it produces**: TCSPC decay histograms from TTTR files.

**Register**: After histogram generation:

```python
register_result(
    kind="processed_data",
    data={"x_ns": time_axis.tolist(), "y_counts": histogram.tolist()},
    parent_artifact_id=tttr_artifact_id,
    sample_id=sample_id,
    operation_type="histogram_construction",
    data_format="json",
    metadata={"channels": selected_channels, "coarsening": coarsening_factor},
)
```

### 7. fcs_correlator (FCS Correlation)

**Plugin path**: `chisurf/plugins/fcs/fcs_correlator/`

**What it produces**: Correlation functions.

**Register**: After correlation computation:

```python
register_result(
    kind="correlation_data",
    data={"tau": tau_array.tolist(), "G": correlation_array.tolist()},
    parent_artifact_id=tttr_artifact_id,
    sample_id=sample_id,
    operation_type="correlation",
    metadata={"channel_pair": [ch1, ch2], "method": correlation_method},
)
```

### 8. jordi_g_factor (G-Factor Calculator)

**Plugin path**: `chisurf/plugins/jordi_g_factor/`

**Covered in PRD-05** but listed here for completeness. Register g-factor as calibration.

---

## Tier 2 Plugins

### 9. pch (Photon Counting Histogram)

**Plugin path**: `chisurf/plugins/pch/`

**Register**: After PCH fit:

```python
register_fit_result(
    fit_data={"brightness": brightness, "N": n_molecules},
    parent_artifact_id=tttr_artifact_id,
    sample_id=sample_id,
    parameters={"brightness": brightness, "N": n_molecules, "chi2": chi2},
)
```

### 10. fcs_calculator (Diffusion/Volume Calculator)

**Plugin path**: `chisurf/plugins/fcs/fcs_calculator/`

**Register**: After calculation:

```python
register_result(
    kind="calibration_data",
    data={"D": diffusion_coeff, "r_h": hydrodynamic_radius, "V_eff": effective_volume},
    operation_type="calibration",
    parameters={"D": diffusion_coeff, "r_h": hydrodynamic_radius},
    metadata={"calibration_type": "confocal_volume"},
)
```

### 11. burst_bva (Burst Variance Analysis)

**Plugin path**: `chisurf/plugins/burst/burst_bva/`

**Register**: After BVA computation:

```python
register_result(
    kind="processed_data",
    data={"E_mean": E_values.tolist(), "sigma_E": sigma_values.tolist()},
    parent_artifact_id=burst_artifact_id,
    sample_id=sample_id,
    operation_type="processing",
    metadata={"analysis_type": "BVA", "window_size": window_size},
)
```

### 12. intensity_trace (Intensity Trace with HMM)

**Plugin path**: `chisurf/plugins/tttr/intensity_trace/`

**Register**: After HMM state detection:

```python
register_result(
    kind="processed_data",
    data={"states": state_sequence.tolist(), "dwell_times": dwell_times.tolist()},
    parent_artifact_id=tttr_artifact_id,
    sample_id=sample_id,
    operation_type="processing",
    metadata={"n_states": n_states, "method": "HMM"},
)
```

### 13. clsm (CLSM Image Analysis)

**Plugin path**: `chisurf/plugins/microscopy/clsm/`

**Register**: After image analysis / ROI selection:

```python
register_result(
    kind="image_data",
    data=image_path_or_array,
    parent_artifact_id=tttr_artifact_id,
    sample_id=sample_id,
    operation_type="image_analysis",
    metadata={"image_type": "intensity", "roi": roi_coords},
)
```

---

## Tier 3 Plugins (Future)

These are lower priority. Add `register_result()` when someone works on them:

- `fret_calculator` -- register computed FRET parameters
- `kappa2_dist` -- register kappa2 distributions
- `fret` (modelling) -- register AV-computed distance distributions
- `fcs_2d` -- register 2D-FLCS matrices
- `burst_mle_analysis` -- register burstwise MLE fits
- `burst_fcs_correlator` -- register burstwise FCS
- `psf_determination` -- register PSF parameters
- `quenching_estimator` -- register quenching rates

---

## Implementation Guide for Each Plugin

Follow these exact steps for every plugin. Do not skip any step.

### Step 1: Read the Plugin

Open the plugin directory. Read `__init__.py` to understand what it does. Find the
main calculation/output function.

### Step 2: Find the Output Point

Look for one of these patterns:
- A function that returns results (return value)
- A function that writes to a file (file path variable)
- A function that updates a GUI widget with results (widget.setText, plot.setData)
- A signal emission (self.result_ready.emit)

### Step 3: Add the Registration Call

Right after the output point, add:

```python
try:
    from chisurf.core.mfdb.result_registry import register_result
    register_result(
        kind="...",          # see ARCHITECTURE.md for valid kinds
        data=...,            # the result data
        sample_id=...,       # from dataset metadata
        parent_artifact_id=...,  # from dataset metadata
        operation_type="...",    # see ARCHITECTURE.md for valid types
        parameters={...},       # numeric results
        metadata={...},         # non-numeric context
    )
except Exception:
    import logging
    logging.getLogger(__name__).debug(
        "MFDB registration skipped", exc_info=True
    )
```

### Step 4: Propagate sample_id

If the plugin has a GUI:
1. Add a `SamplePicker` widget (from PRD-02)
2. Pass the selected `sample_id` to the calculation function

If the plugin is headless (CLI):
1. Add a `--sample-id` argument
2. Pass it through to the registration call

If the plugin works on data from another plugin (e.g., BVA works on burst data):
1. Read the `sample_id` from the input data's metadata
2. Pass it through

### Step 5: Test

For each plugin, add a test that:
1. Creates a temporary MFDB database
2. Runs the plugin's main function with test data
3. Verifies that an artifact was registered in MFDB
4. Verifies the provenance edge exists (if parent_artifact_id was set)

```python
def test_plugin_registers_result(db, tmp_path):
    # Set up test data
    ...

    # Run the plugin function
    result = plugin_function(input_data, db=db)

    # Check artifact exists
    rows = db.con.execute(
        "SELECT COUNT(*) FROM mfdb_artifact WHERE artifact_kind = ?",
        (expected_kind,)
    ).fetchone()
    assert rows[0] >= 1
```

## Definition of Done

- [ ] All Tier 1 plugins (7 plugins) register results in MFDB
- [ ] All Tier 1 plugins have sample_id propagation
- [ ] All Tier 1 plugins have tests for MFDB registration
- [ ] At least 3 Tier 2 plugins register results in MFDB
- [ ] mfdb-admin shows results from all integrated plugins
