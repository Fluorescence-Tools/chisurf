# IRF Estimation - ChiSurf Integration

## Overview

The **IRF Estimation** module provides blind instrument response function (IRF) inference from fluorescence decay data without requiring separate IRF measurements. This implementation uses truncated exponential fitting and Richardson-Lucy deconvolution, reimplemented using numpy/scipy to avoid external dependencies like PyTorch.

There are two main ways to use IRF estimation in ChiSurf:

- **Programmatic API** via the `IRFEstimator` class (this module)
- **GUI plugin** via the *IRF Estimator* plugin (`Fluorescence decay:IRF Estimator`), see `chisurf/plugins/irf_estimator/README.md`

### Reference

This implementation is based on the algorithm described in:

**Adrián Gómez-Sánchez et al., "Blind instrument response function identification from fluorescence decays", _Biophysical Reports_, 2024.**  
DOI: [10.1016/j.bpr.2024.100155](https://doi.org/10.1016/j.bpr.2024.100155)

## Location

Core implementation:

```
chisurf/fluorescence/tcspc/irf_estimation.py
```

GUI plugin (front-end for this module):

```
chisurf/plugins/irf_estimator/
```

## Key Features

- **No Torch Dependency**: Pure numpy/scipy implementation
- **Automatic IRF Extraction**: Estimates IRF directly from decay measurements
- **Multi-Channel Support**: Handles multiple detector channels simultaneously
- **Richardson-Lucy Deconvolution**: Robust deconvolution with regularization
- **Truncated Exponential Fitting**: Accurate decay modeling
- **Visualization Tools**: Built-in plotting functions for validation

## Algorithm Pipeline

1. **Decay Boundary Detection**: Uses Savitzky-Golay filtering to identify decay start (t0) and end (t1)
2. **Exponential Fitting**: Fits truncated exponential model to decay region
3. **Kernel Generation**: Creates normalized deconvolution kernel from fitted parameters
4. **Richardson-Lucy Deconvolution**: Iteratively deconvolves IRF from measured data
5. **Regularization**: Applies median filtering for noise reduction

## Usage

### Basic Usage

```python
from chisurf.fluorescence.tcspc import IRFEstimator
import numpy as np

# Load or create your decay data
time = np.linspace(0, 50, 500)
dt = time[1] - time[0]
decay_data = your_measured_decay  # shape: (n_time, n_channels)

# Estimate IRF
estimator = IRFEstimator(decay_data, dt=dt)
irf = estimator.run()  # Uses default 500 RL iterations

# Access results
print(f"Estimated lifetime: {1.0/estimator.params['k']:.2f} ns")
print(f"Background offset: {estimator.params['C'][0]:.2f}")
```

### Step-by-Step Usage

```python
from chisurf.fluorescence.tcspc import IRFEstimator

# Create estimator
estimator = IRFEstimator(data, dt=0.1)

# Step 1: Find decay boundaries
estimator.find_t0_t1(window_length=11, polyorder=3)

# Step 2: Fit exponential decay
estimator.fit_exponential(method='L-BFGS-B', max_iter=1000)

# Step 3: Generate fitted curves
estimator.generate_data_fit()

# Step 4: Generate deconvolution kernel
estimator.generate_kernel()

# Step 5: Perform Richardson-Lucy deconvolution
estimator.richardson_lucy_deconvolution(iterations=30, regularization=3)

# Access IRF
irf = estimator.irf
```

### Multi-Channel Data

```python
# Data shape: (n_time, n_channels)
multi_channel_data = np.column_stack([decay1, decay2, decay3])

estimator = IRFEstimator(multi_channel_data, dt=0.1)
irf = estimator.run()

# IRF will have same shape as input: (n_time, n_channels)
```

### Visualization

```python
# Plot raw data and exponential fit
fig1, ax1 = estimator.plot_raw_and_fit()

# Plot forward model validation
fig2, ax2 = estimator.plot_forward_model()

import matplotlib.pyplot as plt
plt.show()
```

## API Reference

### IRFEstimator Class

#### Constructor

```python
IRFEstimator(data, dt=1.0)
```

**Parameters:**
- `data` (np.ndarray): 1D or 2D array of fluorescence decay data
  - 1D: shape (n_time,) - single channel
  - 2D: shape (n_time, n_channels) - multiple channels
- `dt` (float): Time step between samples (default: 1.0)

#### Methods

##### `find_t0_t1(window_length=11, polyorder=3, persistence=5, threshold=0.05)`

Estimate decay start (t0) and end (t1) indices using Savitzky-Golay filtering.

**Parameters:**
- `window_length` (int): SG filter window length (must be odd)
- `polyorder` (int): Polynomial order for SG filter
- `persistence` (int): Number of consecutive positive derivative samples for t1
- `threshold` (float): Minimum amplitude threshold for t1 (fraction of range)

##### `fit_exponential(offset=0, method='L-BFGS-B', max_iter=1000)`

Fit truncated exponential curves to decay data.

**Parameters:**
- `offset` (int): Shift applied to t0 when selecting fitting data
- `method` (str): Scipy optimization method
- `max_iter` (int): Maximum optimization iterations

##### `generate_data_fit()`

Generate fitted exponential curves using estimated parameters.

##### `generate_kernel()`

Build normalized deconvolution kernel from fitted exponential.

##### `richardson_lucy_deconvolution(iterations=500, eps=1e-4, regularization=3)`

Perform Richardson-Lucy deconvolution.

**Parameters:**
- `iterations` (int): Number of RL iterations (default: 500, range: 5-2000)
- `eps` (float): Small value to avoid division by zero
- `regularization` (int): Median filter window size (default: 3, set to 1 to disable)

##### `run(**kwargs)`

Execute full IRF estimation pipeline.

**Parameters:** Accepts all parameters from the individual methods above.

**Returns:** np.ndarray - Estimated IRF (shape: [n_time, n_channels])

##### `plot_raw_and_fit(ax=None)`

Plot raw data and fitted exponential curves.

**Returns:** (fig, ax) tuple

##### `plot_forward_model(ax=None)`

Plot forward model (IRF ⊗ exponential) vs measured data.

**Returns:** (fig, ax) tuple

#### Attributes

- `data` (np.ndarray): Input decay data
- `time` (np.ndarray): Time vector
- `dt` (float): Time step
- `num_samples` (int): Number of time points
- `num_channels` (int): Number of channels
- `t0` (np.ndarray): Per-channel decay start indices
- `t1` (np.ndarray): Per-channel decay end indices
- `params` (dict): Fitted parameters {"A", "C", "k"}
  - `A`: Amplitude (per channel)
  - `C`: Constant offset (per channel)
  - `k`: Decay rate (shared across channels)
- `data_fit` (np.ndarray): Fitted exponential curves
- `kernel` (np.ndarray): Deconvolution kernel
- `irf` (np.ndarray): Estimated IRF


## Utility Functions

### `generate_truncated_exponential(t, params)`

Generate truncated exponential curve.

**Parameters:**
- `t` (np.ndarray): Time points
- `params` (dict): {"A", "k", "C", "t0"}

**Returns:** np.ndarray - Model values

### `estimate_lifetime(x, y, t0, t1)`

Estimate lifetime from centroid of decay region.

**Parameters:**
- `x` (np.ndarray): Time points
- `y` (np.ndarray): Signal values
- `t0` (int): Start index
- `t1` (int): End index

**Returns:** float - Estimated lifetime

### `median_filter_nd(x, window_size=3, axes=None, mode='reflect')`

Apply N-dimensional median filter.

### `pad_array(x, pad_left, pad_right, axis, mode='reflect')`

Pad array along one axis.

### `partial_convolution_fft(signal, kernel, axis=0)`

Perform FFT-based convolution along specified axis.

## Examples

See `test/test_irf_estimation.py` for a comprehensive, automated example suite including:
- Basic IRF estimation
- Step-by-step pipeline tests
- Multi-channel data

For GUI-based IRF estimation inside ChiSurf, see the *IRF Estimator Plugin* documentation:

- `chisurf/plugins/irf_estimator/README.md`

## Implementation Notes

### Design Choices

| Aspect | Implementation |
|--------|---------------|
| Backend | NumPy/SciPy |
| GPU Support | No (CPU only) |
| Dependencies | numpy, scipy, matplotlib (optional) |
| Data Format | NumPy arrays |
| Performance | CPU-optimized with FFT-based convolution |

## Technical Details

### Richardson-Lucy Algorithm

The Richardson-Lucy algorithm iteratively refines the IRF estimate:

```
x_{n+1} = x_n * [(y / (x_n ⊗ h)) ⊗ h_reversed]
```

where:
- `x_n` is the current IRF estimate
- `y` is the measured data
- `h` is the deconvolution kernel
- `⊗` denotes convolution

### Regularization

Median filtering is applied after each RL iteration to reduce noise amplification:
- Window size 3 (default) provides good balance
- Set to 1 to disable regularization
- Larger windows provide more smoothing but may over-regularize

### Optimization

The exponential fitting uses scipy.optimize.minimize with:
- Method: L-BFGS-B (bounded optimization)
- Bounds: A > 0, C ≥ 0, k > 0
- Loss: Mean squared error across all channels

## Limitations

1. **Assumes Exponential Decay**: Works best with single or multi-exponential decays
2. **CPU Only**: No GPU acceleration (unlike original torch version)
3. **Shared Decay Rate**: All channels share the same decay rate k
4. **Requires Clear Decay**: Needs sufficient signal-to-noise ratio

## References

### Primary Reference

**Gómez-Sánchez, A., Fersini, F., Zappone, S., Slenders, E., Donato, M., Pelicci, S., Tortarolo, G., Bega, G., Bouzin, M., Cardarelli, F., Lanzanò, L., Koho, S. V., & Vicidomini, G. (2024).** "Blind instrument response function identification from fluorescence decays." _Biophysical Reports_, 4(2), 100155.  
DOI: [10.1016/j.bpr.2024.100155](https://doi.org/10.1016/j.bpr.2024.100155)

### Algorithm References

- **Richardson, W. H. (1972).** "Bayesian-Based Iterative Method of Image Restoration." _Journal of the Optical Society of America_, 62(1), 55-59.
- **Lucy, L. B. (1974).** "An iterative technique for the rectification of observed distributions." _The Astronomical Journal_, 79, 745-754.
- **Savitzky, A.; Golay, M. J. E. (1964).** "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." _Analytical Chemistry_, 36(8), 1627-1639.

## Testing

Run tests with:

```bash
python test/test_irf_estimation.py
```

Or with pytest:

```bash
pytest test/test_irf_estimation.py -v
```

## Support

For issues or questions:
- ChiSurf GitHub: https://github.com/fluorescence-tools/chisurf
