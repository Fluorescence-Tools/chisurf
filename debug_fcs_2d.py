#!/usr/bin/env python3
"""
Debug script to identify specific issues in 2D-MEM fitting.
"""

import sys
import numpy as np
import scipy.io as sio
from pathlib import Path

# Add the plugin path for imports
plugin_path = Path(__file__).parent / "chisurf" / "plugins" / "fcs" / "fcs_2d"
sys.path.insert(0, str(plugin_path.parent.parent.parent))

from chisurf.plugins.fcs.fcs_2d.core import TwoDFDCreator
from chisurf.plugins.fcs.fcs_2d.fitting import TwoDMEMFitter

def debug_fitting():
    """Debug the fitting process with simple synthetic data."""
    print("Creating simple test case for debugging...")
    
    # Create simple synthetic 2D-FDC data
    n_time = 50  # Smaller size for debugging
    n_tau = 10   # Smaller tau range
    
    # Create simple time axis
    time_axis = np.linspace(0, 10, n_time)
    tau_values = np.linspace(0.1, 5.0, n_tau)
    
    # Create simple synthetic 2D-FDC matrix
    # Use a simple exponential decay pattern
    mat_2dfdc = np.zeros((n_time, n_time))
    for i in range(n_time):
        for j in range(n_time):
            if i <= j:  # Make it upper triangular like correlation
                mat_2dfdc[i, j] = 100 * np.exp(-0.5 * (time_axis[i] + time_axis[j]))
    
    # Make it symmetric
    mat_2dfdc = mat_2dfdc + mat_2dfdc.T
    mat_2dfdc_cor = mat_2dfdc.copy()
    
    print(f"Created synthetic data: {mat_2dfdc.shape}")
    print(f"Data range: {mat_2dfdc.min():.2f} to {mat_2dfdc.max():.2f}")
    
    # Create exponential curves
    exp_curve = np.zeros((n_time, n_tau))
    for i, tau in enumerate(tau_values):
        exp_curve[:, i] = np.exp(-time_axis / tau)
    
    print(f"ExpCurve shape: {exp_curve.shape}")
    
    # Test the fitting
    fitter = TwoDMEMFitter()
    
    try:
        print("\nTesting 2D-MEM fitting with synthetic data...")
        result = fitter.fit_2d_mem_wrapper(
            mat_2dfdc=mat_2dfdc,
            mat_2dfdc_cor=mat_2dfdc_cor,
            mat_2dfdc_it=time_axis,
            n_components=2,
            tau_range=(0.1, 5.0),
            tau_step=0.5,
            regulator=0.1,
            y0_initial=1.0,
            max_iterations=500,  # Increased from 50 to 500
            tolerance=1e-4
        )
        
        print(f"Fitting results:")
        print(f"  Success: {result['success']}")
        print(f"  Message: {result['message']}")
        print(f"  Q-value: {result['q_value']:.6f}")
        print(f"  Chi2: {result['chi2']:.6f}")
        print(f"  Entropy: {result['entropy']:.6f}")
        print(f"  Iterations: {result['nit']}")
        
        if result['success']:
            print("✓ Synthetic test: SUCCESS")
        else:
            print("✗ Synthetic test: FAILED")
            
        return result
        
    except Exception as e:
        print(f"Error in synthetic fitting test: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_objective_function():
    """Debug the objective function calculation."""
    print("\n" + "="*50)
    print("DEBUGGING OBJECTIVE FUNCTION")
    print("="*50)
    
    # Create minimal test case
    n_time, n_tau, n_states = 5, 3, 2
    
    # Simple test matrices
    mat_a = np.ones((n_tau, n_states)) * 0.1
    mat_g = np.eye(n_states)
    y0 = 1.0
    
    # Simple data
    mat_2dfdc = np.ones((n_time, n_time))
    mat_2dfdc_cor = mat_2dfdc.copy()
    mat_2dfdc_it = np.arange(n_time)
    
    # Simple exp curve
    exp_curve = np.ones((n_time, n_tau))
    
    # Simple tau values
    tau_values = np.array([1.0, 2.0, 3.0])
    
    # Simple MI matrix
    mi_matrix = np.ones((n_tau, n_states))
    
    # Test differential matrix
    fitter = TwoDMEMFitter()
    diff_mat = fitter._prepare_diff_matrix(mat_2dfdc_it)
    print(f"Differential matrix shape: {diff_mat.shape}")
    print(f"Differential matrix range: {diff_mat.min():.3f} to {diff_mat.max():.3f}")
    
    # Test model calculation
    mat_m_2dflc = mat_a @ mat_g @ mat_a.T
    print(f"Mat_M_2DFLC shape: {mat_m_2dflc.shape}")
    
    try:
        mat_model = exp_curve @ mat_m_2dflc @ exp_curve.T + y0 * diff_mat
        print(f"Model matrix shape: {mat_model.shape}")
        print(f"Model matrix range: {mat_model.min():.3f} to {mat_model.max():.3f}")
        print(f"Data matrix range: {mat_2dfdc_cor.min():.3f} to {mat_2dfdc_cor.max():.3f}")
        
        # Test chi-squared calculation
        chi2 = fitter._calculate_chi2(mat_model, mat_2dfdc_cor, mat_2dfdc)
        print(f"Chi-squared: {chi2:.6f}")
        
        # Test entropy calculation
        entropy = fitter._calculate_entropy(mat_a, mi_matrix)
        print(f"Entropy: {entropy:.6f}")
        
        # Test Q-value
        q_value = chi2 - 2 * entropy / 0.1
        print(f"Q-value: {q_value:.6f}")
        
        print("✓ Objective function test: SUCCESS")
        
    except Exception as e:
        print(f"Error in objective function test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main debug function."""
    print("2D-FLCS Debug Script")
    print("="*50)
    
    # Test objective function
    debug_objective_function()
    
    # Test synthetic fitting
    result = debug_fitting()
    
    print("\n" + "="*50)
    print("DEBUG SUMMARY")
    print("="*50)
    
    if result is not None:
        if result['success']:
            print("✓ Fitting algorithm appears to work")
        else:
            print("✗ Fitting algorithm has convergence issues")
            print("  Possible causes:")
            print("  - Poor initial parameter values")
            print("  - Inappropriate optimization settings")
            print("  - Numerical instabilities in objective function")
    else:
        print("✗ Fitting algorithm has fundamental errors")

if __name__ == "__main__":
    main()
