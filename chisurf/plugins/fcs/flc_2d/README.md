# 2D-FLCS Plugin for ChiSurf

This plugin implements 2D-Fluorescence Lifetime Correlation Spectroscopy (2D-FLCS) analysis for ChiSurf, translating the MATLAB code from the 2D-FLC-code directory into a Python implementation.

## Features

- **2D-FDC Creation**: Generate 2D-Fluorescence Decay Correlation matrices from TTTR data
- **2D-MEM Fitting**: Maximum Entropy Method fitting for multi-exponential decay analysis
- **Interactive GUI**: User-friendly wizard for data loading, parameter setting, and results visualization
- **Data Import/Export**: Support for various TTTR formats and result export
- **Visualization**: Real-time plotting of 2D-FDC matrices and fitting results

## Installation

The plugin is integrated into the ChiSurf plugin system. No additional installation is required beyond the standard ChiSurf setup.

## Usage

### Basic Workflow

1. **Load TTTR Data**: Use the Data Input tab to load your time-tagged time-resolved data
2. **Create 2D-FDC**: Set correlation parameters and generate 2D-FDC matrices
3. **Run MEM Fitting**: Configure fitting parameters and run the Maximum Entropy Method analysis
4. **View Results**: Examine fitting results, amplitude distributions, and transition matrices

### Supported File Formats

- PicoQuant `.pt3`, `.ptu` files
- Becker & Hickl `.hdf5` files
- Binary `.bin` files

## Algorithm Details

### 2D-FDC Creation

The 2D-FDC (Fluorescence Decay Correlation) matrix is created by correlating photon arrival times:

```
G(τ₁, τ₂, Δt) = ⟨δ(t - t₁)δ(t + Δt - t₂)⟩
```

Where:
- τ₁, τ₂ are micro times (fluorescence decay times)
- Δt is the correlation delay time
- t₁, t₂ are macro times (absolute arrival times)

### 2D-MEM Fitting

The Maximum Entropy Method fitting minimizes the Q-function:

```
Q = χ² - 2αS
```

Where:
- χ² is the chi-squared statistic
- S is the entropy
- α is the regularization constant

The model is constructed as:

```
M = E × A × G × Aᵀ × Eᵀ + y₀ × D
```

Where:
- E is the exponential curve matrix
- A is the amplitude matrix
- G is the transition matrix
- y₀ is the baseline
- D is the differential matrix

## Parameters

### 2D-FDC Parameters

- **dT**: Correlation delay time (seconds)
- **ddT**: Correlation delay window width (seconds)
- **Tstart/Tend**: Analysis time window (seconds)
- **tMin/tMax**: Micro time range (nanoseconds)
- **tStep**: Time bin size (nanoseconds)
- **lint_bin_factor**: Linear binning factor
- **logt_imax**: Number of logarithmic time points

### MEM Fitting Parameters

- **Number of States**: Number of kinetic states to fit
- **Tau Range**: Lifetime range for components (nanoseconds)
- **Regulator Const**: Regularization constant for entropy
- **Initial y0**: Baseline intensity value

## MATLAB Code Translation

This plugin is based on the MATLAB implementation from the 2D-FLC-code directory:

- `TK_Create2DFDC_04.m` → `core.TwoDFDCreator.create_2d_fdc()`
- `TK_FitF_2DMEM_07.m` → `fitting.TwoDMEMFitter.fit_2d_mem()`
- `TK_MyMain_Fit_2DMEM_04.m` → `wizard.TwoDFCSWizard`

The Python implementation maintains the same mathematical algorithms while providing:

- Improved error handling and validation
- Progress tracking for long calculations
- Modern GUI with interactive plots
- Integration with ChiSurf data structures

## Testing

Run the test suite to verify functionality:

```bash
cd chisurf/plugins/fcs/fcs_2d
python test_2dfcs.py
```

The test suite includes:
- 2D-FDC matrix creation verification
- 2D-MEM fitting algorithm validation
- Utility function testing

## Dependencies

- NumPy
- SciPy
- PyQt5/PyQt2
- PyQtGraph
- h5py (for HDF5 support)

## API Reference

### TwoDFDCreator

```python
creator = TwoDFDCreator()
mat_lin, mat_lin_t, mat_log, mat_log_t = creator.create_2d_fdc(
    macro_times, micro_times, dT=0.1, ddT=0.05, ...
)
```

### TwoDMEMFitter

```python
fitter = TwoDMEMFitter()
result = fitter.fit_2d_mem(
    initial_mat_a, initial_mat_g, initial_y0,
    fix_mat_a, fix_mat_g, fix_y0, ...
)
```

### Utility Functions

```python
# Load TTTR data
macro_times, micro_times, metadata = load_tttr_data(file_path)

# Create exponential curves
exp_curves = create_exponential_curve(tau_values, time_axis)

# Create MI matrix
mi_matrix = create_mi_matrix(tau_values, n_states, method='gaussian')
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Large datasets may require reducing the number of time bins or using a smaller correlation window
2. **Convergence Issues**: Try different initial parameters or adjust the regularization constant
3. **File Loading**: Ensure TTTR files are in supported formats and not corrupted

### Performance Tips

- Use appropriate time ranges to limit matrix sizes
- Start with fewer states and increase gradually
- Monitor memory usage for large datasets

## Contributing

To extend the plugin:

1. Add new TTTR format support in `utils.load_tttr_data()`
2. Implement additional fitting algorithms in `fitting.py`
3. Add new visualization options in `wizard.py`
4. Extend the test suite with new test cases

## References

- Original MATLAB implementation: 2D-FLC-code directory
- 2D-FLCS methodology literature
- Maximum Entropy Method theory

## License

This plugin is part of the ChiSurf project and follows the same licensing terms.
