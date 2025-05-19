"""
Test script for lltf module.

This script tests the lltf module by fitting a lifetime to the example data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import lltf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lltf import fit_lifetime, Decay

def test_fit_lifetime():
    """Test the fit_lifetime function."""
    print("Testing fit_lifetime function...")
    
    # Fit a lifetime to the example data
    result = fit_lifetime(
        decay_file='example/5-44_D0.dat',
        irf_file='example/IRF_D0.dat',
        n_lifetimes=2,
        verbose=True
    )
    
    # Print the results
    print("\nFit results:")
    for i in range(result['n_lifetimes']):
        lifetime = result['lifetime_spectrum'][2*i+1]
        amplitude = result['lifetime_spectrum'][2*i]
        print(f"Lifetime {i+1}: {lifetime:.3f} ns, Amplitude: {amplitude:.3f}")
    
    print(f"IRF shift: {result['irf_shift']:.3f} ns")
    print(f"Decay background: {result['decay_background']:.2f}")
    print(f"Reduced chi-square: {result['reduced_chi_square']:.2f}")
    
    return result

def test_decay_class():
    """Test the Decay class."""
    print("\nTesting Decay class...")
    
    # Load data
    decay_data = np.genfromtxt('example/5-44_D0.dat')
    irf_data = np.genfromtxt('example/IRF_D0.dat')
    
    # Create Decay object
    decay = Decay(
        decay=decay_data[:, 1],
        irf=irf_data[:, 1],
        time_axis=decay_data[:, 0]
    )
    
    # Set analysis range
    decay.get_analysis_range(verbose=True)
    
    # Estimate background
    decay.estimate_background(verbose=True)
    
    # Estimate IRF shift
    decay.estimate_irf_shift(verbose=True)
    
    # Fit with 2 lifetime components
    result = decay.fit(n_lifetimes=2, verbose=True)
    
    # Plot the results
    decay.plot()
    
    return result

if __name__ == "__main__":
    # Test the fit_lifetime function
    test_fit_lifetime()
    
    # Test the Decay class
    test_decay_class()