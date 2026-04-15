#!/usr/bin/env python3

"""
Simplified script to perform joint fitting of TCSPC data from the ibh_sample folder.
This follows the pattern from the existing test.
"""

import pathlib
import sys

# Add the chisurf package to the path
chisurf_path = pathlib.Path(__file__).parent
sys.path.insert(0, str(chisurf_path))

import chisurf.experiments
import chisurf.models
import chisurf.fitting

def main():
    print("Starting simplified joint TCSPC fitting...")
    
    # Set up the experiment and reader (same as in the test)
    dt = 0.0141
    tcspc_experiment = chisurf.experiments.core.Experiment(name='TCSPC')
    tcspc_reader = chisurf.experiments.tcspc.TCSPCReader(
        is_jordi=False,
        skiprows=10,
        dt=dt,
        experiment=tcspc_experiment
    )
    
    # Load the data files (same as in the test)
    base_path = pathlib.Path('./test/data/tcspc/ibh_sample')
    
    print("Loading IRF...")
    irf = tcspc_reader.read(filename=str(base_path / 'Prompt.txt'))
    
    print("Loading decay data...")
    decay_dd_d0 = tcspc_reader.read(filename=str(base_path / 'Decay_577D.txt'))
    decay_dd_da = tcspc_reader.read(filename=str(base_path / 'Decay_577D+577A+GTPgS.txt'))
    
    # Create fits exactly as in the test
    print("Creating donor-only fit...")
    fit_d0 = chisurf.fitting.fit.FitGroup(
        data=decay_dd_d0,
        model_class=chisurf.models.tcspc.lifetime.LifetimeModel
    )
    
    model_d0 = fit_d0.model
    model_d0.lifetimes.append()
    model_d0.convolve._irf = irf[0]
    model_d0.update()
    fit_d0.model.find_parameters()
    fit_d0.fit_range = 0, 2000
    
    print("Creating FRET fit...")
    fit_da = chisurf.fitting.fit.FitGroup(
        data=decay_dd_da,
        model_class=chisurf.models.tcspc.fret.GaussianModel
    )
    
    model_da = fit_da.model
    model_da.append(mean=50, sigma=6, species_fraction=1.0)
    model_da.find_parameters()
    
    # Link the donor lifetime parameter between the two fits
    # This creates the joint fitting scenario
    model_da.parameter_dict['tL1'].link = model_d0.parameter_dict['tL1']
    fit_da.fit_range = 0, 2000
    
    print("Running donor-only fit...")
    chi2_d0_before = fit_d0.chi2
    fit_d0.run()
    chi2_d0_after = fit_d0.chi2
    print(f"Donor fit - Chi² before: {chi2_d0_before:.2f}, after: {chi2_d0_after:.2f}")
    
    print("Running FRET fit...")
    chi2_da_before = fit_da.chi2
    fit_da.run()
    chi2_da_after = fit_da.chi2
    print(f"FRET fit - Chi² before: {chi2_da_before:.2f}, after: {chi2_da_after:.2f}")
    
    # Print fitted parameters
    print("\nDonor-only fitted parameters:")
    for param_name in model_d0.parameter_names:
        param = model_d0.parameter_dict[param_name]
        print(f"  {param_name}: {param.value:.4f} ± {param.stdev:.4f}")
    
    print("\nFRET fitted parameters:")
    for param_name in model_da.parameter_names:
        param = model_da.parameter_dict[param_name]
        linked_info = " (linked)" if param.link is not None else ""
        print(f"  {param_name}: {param.value:.4f} ± {param.stdev:.4f}{linked_info}")
    
    # Calculate FRET efficiency
    if 'E_FRET' in model_da.parameter_names:
        fret_efficiency = model_da.parameter_dict['E_FRET'].value
        print(f"\nFRET efficiency: {fret_efficiency:.3f}")
    
    print("Joint fitting completed successfully!")

if __name__ == "__main__":
    main()