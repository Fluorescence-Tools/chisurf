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
    from chisurf.core.fluorescence.tcspc import IRFEstimator
    
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
    from chisurf.core.fluorescence.tcspc import IRFEstimator
    
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
    from chisurf.core.fluorescence.tcspc import IRFEstimator
    
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
    from chisurf.core.fluorescence.tcspc import IRFEstimator
    
    assert IRFEstimator is not None
    assert hasattr(IRFEstimator, 'run')


def test_utility_functions():
    """Test utility functions"""
    from chisurf.core.fluorescence.tcspc.irf_estimation import (
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
    from chisurf.core.fluorescence.tcspc import IRFEstimator
    
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


if __name__ == '__main__':
    # Run tests
    test_irf_estimator_basic()
    print("✓ Basic IRF estimation test passed")
    
    test_irf_estimator_run()
    print("✓ Full pipeline test passed")
    
    test_irf_estimator_multi_channel()
    print("✓ Multi-channel test passed")
    
    test_irf_estimator_import()
    print("✓ IRFEstimator import test passed")
    
    test_utility_functions()
    print("✓ Utility functions test passed")
    
    test_error_handling()
    print("✓ Error handling test passed")
    
    print("\nAll tests passed!")
