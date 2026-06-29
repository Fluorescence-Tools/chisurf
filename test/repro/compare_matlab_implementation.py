#!/usr/bin/env python3
"""
Compare Python 2D-FLCS implementation with MATLAB reference to identify convergence issues.

This script analyzes the differences between our Python implementation and the MATLAB reference
to identify why convergence is poor.
"""

import sys
import numpy as np
import scipy.io as sio
from pathlib import Path

# Add the plugin path for imports
plugin_path = Path(__file__).parent.parent / "chisurf" / "plugins" / "fcs" / "fcs_2d"
sys.path.insert(0, str(plugin_path.parent.parent.parent))

from chisurf.plugins.fcs.flc_2d.core import TwoDFDCreator
from chisurf.plugins.fcs.flc_2d.fitting import TwoDMEMFitter

def load_matlab_data():
    """Load MATLAB reference data."""
    print("Loading MATLAB reference data...")
    
    # Load simulated data
    data_path = Path(__file__).parent / "playground" / "2D-FLC-code" / "simulated_data.mat"
    if not data_path.exists():
        raise FileNotFoundError(f"MATLAB data file not found: {data_path}")
    
    data = sio.loadmat(str(data_path))
    
    # Extract key arrays and parameters
    tt1 = data['tt1'].flatten()  # macro times
    kin1 = data['kin1'].flatten()  # micro times
    num_of_state = int(data['NumOfState'][0, 0])
    life = data['Life'].flatten()  # lifetimes
    trans_rate = data['TransRate']  # transition rate matrix
    tstep = data['Tstep'][0, 0]  # time step in seconds
    dwell_tstep = data['Dwell_Tstep'][0, 0]  # dwell time step
    
    print(f"Loaded {len(tt1)} photons")
    print(f"Number of states: {num_of_state}")
    print(f"Lifetimes: {life} ns")
    print(f"Time step: {tstep*1e9:.3f} ns")
    print(f"Dwell time step: {dwell_tstep*1e6:.1f} μs")
    print(f"Macro time range: {tt1[0]:.2e} to {tt1[-1]:.2e}")
    print(f"Micro time range: {kin1.min():.1f} to {kin1.max():.1f}")
    
    return {
        'tt1': tt1,
        'kin1': kin1,
        'num_of_state': num_of_state,
        'life': life,
        'trans_rate': trans_rate,
        'tstep': tstep,
        'dwell_tstep': dwell_tstep
    }

def analyze_matlab_parameters():
    """Analyze MATLAB reference parameters."""
    print("\n" + "="*60)
    print("ANALYZING MATLAB REFERENCE PARAMETERS")
    print("="*60)
    
    # Read MATLAB reference code to extract parameters
    matlab_files = [
        "TK_Create2DFDC_04.m",
        "TK_FitF_2DMEM_07.m",
        "TK_MyMain_Fit_2DMEM_04.m"
    ]
    
    matlab_dir = Path(__file__).parent / "playground" / "2D-FLC-code" / "MatlabCodes"
    
    for mfile in matlab_files:
        filepath = matlab_dir / mfile
        if filepath.exists():
            print(f"\nAnalyzing {mfile}:")
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Extract key parameters
                if 'dT' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'dT' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = float(val)
                                print(f"  dT = {val_num} seconds = {val_num*1e6:.1f} μs")
                            except:
                                print(f"  dT = {val}")
                
                if 'ddT' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'ddT' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = float(val)
                                print(f"  ddT = {val_num} seconds = {val_num*1e6:.1f} μs")
                            except:
                                print(f"  ddT = {val}")
                
                if 'tMin' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'tMin' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = float(val)
                                print(f"  tMin = {val_num} ns")
                            except:
                                print(f"  tMin = {val}")
                
                if 'tMax' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'tMax' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = float(val)
                                print(f"  tMax = {val_num} ns")
                            except:
                                print(f"  tMax = {val}")
                
                if 'tStep' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'tStep' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = float(val)
                                print(f"  tStep = {val_num} ns")
                            except:
                                print(f"  tStep = {val}")
                
                if 'lint_BinFactor' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'lint_BinFactor' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = int(val)
                                print(f"  lint_BinFactor = {val_num}")
                            except:
                                print(f"  lint_BinFactor = {val}")
                
                if 'logt_Imax' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'logt_Imax' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = int(val)
                                print(f"  logt_Imax = {val_num}")
                            except:
                                print(f"  logt_Imax = {val}")
                
                if 'RegulatorConst' in content:
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    for line in lines:
                        if 'RegulatorConst' in line and '=' in line:
                            val = line.split('=')[1].strip().rstrip(';')
                            try:
                                val_num = float(val)
                                print(f"  RegulatorConst = {val_num}")
                            except:
                                print(f"  RegulatorConst = {val}")
                
            except Exception as e:
                print(f"  Error analyzing {mfile}: {e}")
        else:
            print(f"  File not found: {mfile}")

def compare_implementation_differences():
    """Compare Python implementation with MATLAB reference."""
    print("\n" + "="*60)
    print("COMPARING IMPLEMENTATION DIFFERENCES")
    print("="*60)
    
    print("Key differences found:")
    
    print("\n1. PARAMETER SCALING:")
    print("   MATLAB: dT=0.1s, ddT=0.05s")
    print("   Python: dT=100000 ticks, ddT=50000 ticks")
    print("   Issue: We're converting seconds to ticks, but the scale may be wrong")
    
    print("\n2. OPTIMIZATION SETTINGS:")
    print("   MATLAB: fminsearch with MaxFunEvals=10^4")
    print("   Python: minimize with maxiter=100")
    print("   Issue: Much fewer iterations allowed")
    
    print("\n3. INITIAL PARAMETERS:")
    print("   MATLAB: Uses specific initial estimates from 1D fitting")
    print("   Python: Uses generic small positive values")
    print("   Issue: Poor initial guesses")
    
    print("\n4. REGULARIZATION:")
    print("   MATLAB: RegulatorConst = 10^(-1+0) = 0.1")
    print("   Python: Same value but may need tuning")
    print("   Issue: May need different regularization for real data")
    
    print("\n5. CONVERGENCE CRITERIA:")
    print("   MATLAB: Uses xatol and fatol tolerances")
    print("   Python: Uses same but may need adjustment")
    print("   Issue: Tolerances may be too strict/loose")

def create_optimized_test():
    """Create a test with MATLAB-like parameters."""
    print("\n" + "="*60)
    print("CREATING OPTIMIZED TEST WITH MATLAB-LIKE PARAMETERS")
    print("="*60)
    
    # Load MATLAB data
    matlab_data = load_matlab_data()
    
    # Use MATLAB-like parameters
    dT = 0.1  # seconds
    ddT = 0.05  # seconds
    tMin = 1.0  # ns
    tMax = 12.0  # ns
    tStep = 0.004  # ns
    lint_bin_factor = 2
    logt_imax = 100
    
    print(f"Using MATLAB-like parameters:")
    print(f"  dT = {dT*1e6:.1f} μs")
    print(f"  ddT = {ddT*1e6:.1f} μs")
    print(f"  tMin = {tMin} ns")
    print(f"  tMax = {tMax} ns")
    print(f"  tStep = {tStep} ns")
    print(f"  lint_bin_factor = {lint_bin_factor}")
    print(f"  logt_imax = {logt_imax}")
    
    # Convert to tick units
    micro_times = matlab_data['kin1'].astype(np.int64)
    macro_times = (matlab_data['tt1'] / matlab_data['tstep']).astype(np.int64)
    
    dT_ticks = int(dT / matlab_data['tstep'])
    ddT_ticks = int(ddT / matlab_data['tstep'])
    tMin_ticks = int(tMin / tStep)
    tMax_ticks = int(tMax / tStep)
    
    print(f"\nConverted to tick units:")
    print(f"  dT_ticks = {dT_ticks}")
    print(f"  ddT_ticks = {ddT_ticks}")
    print(f"  tMin_ticks = {tMin_ticks}")
    print(f"  tMax_ticks = {tMax_ticks}")
    print(f"  macro_time range: {macro_times.min()} to {macro_times.max()}")
    print(f"  micro_time range: {micro_times.min()} to {micro_times.max()}")
    
    # Create Python 2D-FDC
    print("\nCreating 2D-FDC with MATLAB-like parameters...")
    creator = TwoDFDCreator()
    
    try:
        mat_lin, mat_lin_t, mat_log, mat_log_t = creator.create_2d_fdc(
            macro_times=macro_times,
            micro_times=micro_times,
            dT=dT_ticks,
            ddT=ddT_ticks,
            tMin=tMin_ticks,
            tMax=tMax_ticks,
            logt_imax=logt_imax
        )
        
        print(f"Success! Created matrices:")
        print(f"  Linear matrix: {mat_lin.shape}")
        print(f"  Linear time axis: {mat_lin_t.shape}")
        print(f"  Log matrix: {mat_log.shape}")
        print(f"  Log time axis: {mat_log_t.shape}")
        
        # Check matrix properties
        print(f"\nMatrix statistics:")
        print(f"  Linear matrix - min: {mat_lin.min()}, max: {mat_lin.max()}, sum: {mat_lin.sum()}")
        print(f"  Log matrix - min: {mat_log.min()}, max: {mat_log.max()}, sum: {mat_log.sum()}")
        
        # Check if matrices are non-zero
        if mat_lin.sum() == 0:
            print("  WARNING: Linear matrix is all zeros!")
        if mat_log.sum() == 0:
            print("  WARNING: Log matrix is all zeros!")
        
        return {
            'mat_lin': mat_lin,
            'mat_lin_t': mat_lin_t,
            'mat_log': mat_log,
            'mat_log_t': mat_log_t,
            'parameters': {
                'dT_ticks': dT_ticks,
                'ddT_ticks': ddT_ticks,
                'tMin_ticks': tMin_ticks,
                'tMax_ticks': tMax_ticks,
                'logt_imax': logt_imax
            }
        }
        
    except Exception as e:
        print(f"Error in 2D-FDC creation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_optimized_fitting(fdc_data, matlab_data):
    """Test fitting with optimized parameters."""
    print("\n" + "="*60)
    print("TESTING FITTING WITH OPTIMIZED PARAMETERS")
    print("="*60)
    
    if fdc_data is None:
        print("Skipping 2D-MEM test due to 2D-FDC creation failure")
        return None
    
    # Use MATLAB-like fitting parameters
    tau_min = 0.05  # MATLAB uses 0.05
    tau_max = 5.05   # MATLAB uses 5.05
    tau_step = 0.05  # MATLAB uses 0.05
    regulator = 0.1  # MATLAB uses 0.1
    y0_initial = 1000.0  # Reasonable starting value
    
    print(f"Fitting parameters:")
    print(f"  tau_range: ({tau_min}, {tau_max})")
    print(f"  tau_step: {tau_step}")
    print(f"  regulator: {regulator}")
    print(f"  y0_initial: {y0_initial}")
    print(f"  max_iterations: 5000 (increased from 100)")
    print(f"  tolerance: 1e-6 (tighter tolerance)")
    
    # Create fitter
    fitter = TwoDMEMFitter()
    
    try:
        print("\nTesting 2D-MEM fitting with optimized parameters...")
        result = fitter.fit_2d_mem_wrapper(
            mat_2dfdc=fdc_data['mat_lin'],
            mat_2dfdc_cor=fdc_data['mat_lin'],  # Use same matrix for correlation
            mat_2dfdc_it=fdc_data['mat_lin_t'],
            n_components=matlab_data['num_of_state'],
            tau_range=(tau_min, tau_max),
            tau_step=tau_step,
            regulator=regulator,
            y0_initial=y0_initial,
            max_iterations=5000,  # Much higher for better convergence
            tolerance=1e-6  # Tighter tolerance
        )
        
        print(f"Fitting results:")
        print(f"  Success: {result['success']}")
        print(f"  Message: {result['message']}")
        print(f"  Q-value: {result['q_value']:.6f}")
        print(f"  Chi2: {result['chi2']:.6f}")
        print(f"  Entropy: {result['entropy']:.6f}")
        print(f"  Iterations: {result['nit']}")
        
        # Check convergence quality
        if result['success']:
            print("✓ Optimized fitting: SUCCESS")
        else:
            print("✗ Optimized fitting: FAILED - but with better parameters")
            
        return result
        
    except Exception as e:
        print(f"Error in optimized fitting test: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main comparison function."""
    print("2D-FLCS Implementation Comparison with MATLAB Reference")
    print("="*60)
    
    # Analyze MATLAB parameters
    analyze_matlab_parameters()
    
    # Compare implementation differences
    compare_implementation_differences()
    
    # Load MATLAB data
    try:
        matlab_data = load_matlab_data()
    except Exception as e:
        print(f"Failed to load MATLAB reference data: {e}")
        return
    
    # Test with optimized parameters
    fdc_data = create_optimized_test()
    
    if fdc_data is not None:
        fit_result = test_optimized_fitting(fdc_data, matlab_data)
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("Key findings:")
    print("1. Parameter scaling needs verification")
    print("2. Optimization settings need adjustment")
    print("3. Initial parameters need improvement")
    print("4. Convergence criteria may need tuning")
    print("\nRecommendations:")
    print("- Verify parameter units and scaling")
    print("- Increase max_iterations to 5000+")
    print("- Use tighter tolerance (1e-6)")
    print("- Implement better initial parameter estimation")
    print("- Consider alternative optimization methods")

if __name__ == "__main__":
    main()
