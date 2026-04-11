# Consolidated test file: test_irf.py


# --- FROM test_irf_estimation.py ---
"""
Test script for IRF estimation functionality

Tests the IRFEstimator class which implements blind instrument response function
estimation from fluorescence decay data.

Reference:
    Gómez-Sánchez et al., "Blind instrument response function identification from 
    fluorescence decays", Biophysical Reports, 2024.
    https://doi.org/10.1016/j.bpr.2024.100155
"""
import numpy as np
import pytest


def test_irf_estimator_basic():
    """Test basic IRF estimation functionality"""
    from chisurf.fluorescence.tcspc import IRFEstimator
    
    # Create synthetic decay data
    time = np.linspace(0, 50, 500)
    dt = time[1] - time[0]
    
    # Simulate a decay with known parameters
    tau = 4.0  # lifetime
    irf_width = 0.5
    irf_center = 5.0
    
    # Create a Gaussian IRF
    irf_true = np.exp(-0.5 * ((time - irf_center) / irf_width) ** 2)
    irf_true /= irf_true.sum()
    
    # Create exponential decay
    decay = np.exp(-time / tau)
    
    # Convolve to create measured signal
    measured = np.convolve(decay, irf_true, mode='same')
    measured += 0.1  # Add offset
    measured = measured.reshape(-1, 1)  # Single channel
    
    # Estimate IRF
    estimator = IRFEstimator(measured, dt=dt)
    
    # Test find_t0_t1
    estimator.find_t0_t1()
    assert estimator.t0 is not None
    assert estimator.t1 is not None
    assert len(estimator.t0) == 1
    assert len(estimator.t1) == 1
    
    # Test fit_exponential
    estimator.fit_exponential()
    assert estimator.params is not None
    assert 'A' in estimator.params
    assert 'C' in estimator.params
    assert 'k' in estimator.params
    
    # Test generate_data_fit
    estimator.generate_data_fit()
    assert estimator.data_fit is not None
    assert estimator.data_fit.shape == measured.shape
    
    # Test generate_kernel
    estimator.generate_kernel()
    assert estimator.kernel is not None
    assert len(estimator.kernel) == len(time)
    
    # Test richardson_lucy_deconvolution
    estimator.richardson_lucy_deconvolution(iterations=10)
    assert estimator.irf is not None
    assert estimator.irf.shape == measured.shape


def test_irf_estimator_run():
    """Test full pipeline with run() method"""
    from chisurf.fluorescence.tcspc import IRFEstimator
    
    # Create synthetic decay data
    time = np.linspace(0, 50, 500)
    dt = time[1] - time[0]
    
    # Simple exponential decay with offset
    decay = np.exp(-time / 4.0) + 0.1
    decay = decay.reshape(-1, 1)
    
    # Run full pipeline
    estimator = IRFEstimator(decay, dt=dt)
    irf = estimator.run(rl_iterations=10)
    
    assert irf is not None
    assert irf.shape == decay.shape
    assert np.all(irf >= 0)  # IRF should be positive


def test_irf_estimator_multi_channel():
    """Test IRF estimation with multiple channels"""
    from chisurf.fluorescence.tcspc import IRFEstimator
    
    # Create synthetic decay data with 3 channels
    time = np.linspace(0, 50, 500)
    dt = time[1] - time[0]
    
    # Create 3 channels with slightly different parameters
    decay1 = np.exp(-time / 3.5) + 0.1
    decay2 = np.exp(-time / 4.0) + 0.12
    decay3 = np.exp(-time / 4.5) + 0.08
    
    data = np.column_stack([decay1, decay2, decay3])
    
    # Run estimation
    estimator = IRFEstimator(data, dt=dt)
    irf = estimator.run(rl_iterations=10)
    
    assert irf is not None
    assert irf.shape == data.shape
    assert irf.shape[1] == 3  # 3 channels
    assert np.all(irf >= 0)


def test_irf_estimator_import():
    """Test that IRFEstimator can be imported"""
    from chisurf.fluorescence.tcspc import IRFEstimator
    
    assert IRFEstimator is not None
    assert hasattr(IRFEstimator, 'run')


def test_utility_functions():
    """Test utility functions"""
    from chisurf.fluorescence.tcspc.irf_estimation import (
        pad_array,
        median_filter_nd,
        generate_truncated_exponential,
        estimate_lifetime
    )
    
    # Test pad_array
    x = np.array([1, 2, 3, 4, 5])
    padded = pad_array(x, 2, 2, axis=0, mode='reflect')
    assert len(padded) == 9
    
    # Test median_filter_nd
    x = np.random.randn(100, 3)
    filtered = median_filter_nd(x, window_size=3, axes=[0])
    assert filtered.shape == x.shape
    
    # Test generate_truncated_exponential
    t = np.linspace(0, 10, 100)
    params = {"A": 1.0, "k": 0.5, "C": 0.1, "t0": 2.0}
    y = generate_truncated_exponential(t, params)
    assert len(y) == len(t)
    assert np.all(y[t < 2.0] == 0.1)  # Before t0, should be constant C
    
    # Test estimate_lifetime
    t = np.linspace(0, 20, 200)
    y = np.exp(-t / 4.0)
    tau = estimate_lifetime(t, y, 0, len(t)-1)
    assert 3.0 < tau < 5.0  # Should be close to 4.0


def test_error_handling():
    """Test error handling"""
    from chisurf.fluorescence.tcspc import IRFEstimator
    
    # Test with wrong dimensions
    with pytest.raises(ValueError):
        data = np.random.randn(10, 10, 10)  # 3D array
        estimator = IRFEstimator(data)
    
    # Test calling methods out of order
    data = np.random.randn(100, 1)
    estimator = IRFEstimator(data)
    
    with pytest.raises(RuntimeError):
        estimator.fit_exponential()  # Should fail, need to call find_t0_t1 first
    
    with pytest.raises(RuntimeError):
        estimator.generate_data_fit()  # Should fail, need params first
    
    with pytest.raises(RuntimeError):
        estimator.generate_kernel()  # Should fail, need params first
    
    with pytest.raises(RuntimeError):
        estimator.richardson_lucy_deconvolution()  # Should fail, need kernel first



# --- FROM test_irf_normalization_contract.py ---
from pathlib import Path


def test_irf_is_normalized_before_convolution_paths():
    path = Path(__file__).resolve().parents[2] / "chisurf" / "models" / "tcspc" / "nusiance.py"
    src = path.read_text(encoding="utf-8")

    norm_idx = src.find("irf_y = irf_y / np.sum(irf_y)")
    assert norm_idx != -1

    periodic_idx = src.find("convolve_lifetime_spectrum_periodic_nb", norm_idx)
    exp_idx = src.find("convolve_lifetime_spectrum_nb", norm_idx)
    full_idx = src.find("np.convolve(data, irf_y", norm_idx)

    assert periodic_idx > norm_idx
    assert exp_idx > norm_idx
    assert full_idx > norm_idx

# --- FROM test_irf_truncation.py ---
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the chisurf directory to the path
sys.path.append('/')

import chisurf
from chisurf.curve import Curve
from chisurf.data import DataCurve
from chisurf.fitting.fit import Fit
from chisurf.models.tcspc.lifetime import LifetimeModel

# Create a simple test data
x = np.linspace(0, 10, 100)
y = np.exp(-x) + 0.1 * np.random.randn(len(x))
data = DataCurve(x=x, y=y)

# Create a simple IRF
irf_x = np.linspace(0, 10, 100)
irf_y = np.exp(-((irf_x - 2) ** 2) / 0.5)
irf = Curve(x=irf_x, y=irf_y)

# Create a fit object
fit = Fit(model_class=LifetimeModel, data=data)

# Set the IRF
fit.model.convolve._irf = irf

# Print the initial IRF
print("Initial IRF shape:", fit.model.convolve.irf.y.shape)
print("Initial IRF sum:", np.sum(fit.model.convolve.irf.y))

# Set the IRF truncation parameters
fit.model.convolve.irf_start = 20
fit.model.convolve.irf_stop = 80

# Get the truncated IRF
truncated_irf = fit.model.convolve.irf

# Print the truncated IRF
print("Truncated IRF shape:", truncated_irf.y.shape)
print("Truncated IRF sum:", np.sum(truncated_irf.y))
print("Non-zero values in truncated IRF:", np.count_nonzero(truncated_irf.y))

# Plot the original and truncated IRF
plt.figure(figsize=(10, 6))
plt.plot(irf_x, irf_y, 'b-', label='Original IRF')
plt.plot(truncated_irf.x, truncated_irf.y, 'r-', label='Truncated IRF')
plt.axvline(x=irf_x[20], color='g', linestyle='--', label='irf_start')
plt.axvline(x=irf_x[80], color='m', linestyle='--', label='irf_stop')
plt.legend()
plt.title('Original vs Truncated IRF')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.savefig('irf_truncation_test.png')

# Get the curves from the fit object
curves = fit.get_curves()
print("Curves keys:", curves.keys())

# Check if the IRF in the curves is truncated
if 'IRF' in curves:
    irf_curve = curves['IRF']
    print("IRF curve from get_curves() shape:", irf_curve.y.shape)
    print("IRF curve from get_curves() sum:", np.sum(irf_curve.y))
    print("Non-zero values in IRF curve from get_curves():", np.count_nonzero(irf_curve.y))
    
    # Plot the IRF from get_curves()
    plt.figure(figsize=(10, 6))
    plt.plot(irf_x, irf_y, 'b-', label='Original IRF')
    plt.plot(irf_curve.x, irf_curve.y, 'g-', label='IRF from get_curves()')
    plt.axvline(x=irf_x[20], color='g', linestyle='--', label='irf_start')
    plt.axvline(x=irf_x[80], color='m', linestyle='--', label='irf_stop')
    plt.legend()
    plt.title('Original vs IRF from get_curves()')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.savefig('irf_from_get_curves_test.png')
else:
    print("No IRF curve in get_curves() result")

print("Test completed successfully!")