#!/usr/bin/env python3

"""
Simplified joint fitting using LifetimeModel instead of GaussianModel.
This demonstrates joint fitting without the complex FRET model.
"""

import pathlib
import sys

# Add the chisurf package to the path
chisurf_path = pathlib.Path(__file__).parent
sys.path.insert(0, str(chisurf_path))

import chisurf.core.experiments
import chisurf.core.models
import chisurf.core.fitting

def main():
    print("Starting simplified joint TCSPC fitting with LifetimeModel...")
    
    # Set up the experiment and reader
    dt = 0.0141
    tcspc_experiment = chisurf.core.experiments.core.Experiment(name='TCSPC')
    tcspc_reader = chisurf.core.experiments.tcspc.TCSPCReader(
        is_jordi=False,
        skiprows=10,
        dt=dt,
        experiment=tcspc_experiment
    )
    
    # Load the data files
    base_path = pathlib.Path('./test/data/tcspc/ibh_sample')
    
    print("Loading IRF...")
    irf = tcspc_reader.read(filename=str(base_path / 'Prompt.txt'))
    
    print("Loading decay data...")
    decay_donor_only = tcspc_reader.read(filename=str(base_path / 'Decay_577D.txt'))
    decay_fret = tcspc_reader.read(filename=str(base_path / 'Decay_577D+577A+GTPgS.txt'))
    
    # Create individual fits with LifetimeModel
    print("Creating donor-only fit...")
    fit_donor = chisurf.core.fitting.fit.FitGroup(
        data=decay_donor_only,
        model_class=chisurf.core.models.tcspc.lifetime.LifetimeModel
    )
    
    model_donor = fit_donor.model
    model_donor.lifetimes.append()  # Add one lifetime component
    model_donor.convolve._irf = irf[0]
    model_donor.update()
    model_donor.find_parameters()
    fit_donor.fit_range = 100, 1500
    
    print("Creating FRET decay fit...")
    fit_fret = chisurf.core.fitting.fit.FitGroup(
        data=decay_fret,
        model_class=chisurf.core.models.tcspc.lifetime.LifetimeModel
    )
    
    model_fret = fit_fret.model
    model_fret.lifetimes.append()  # Add one lifetime component
    model_fret.convolve._irf = irf[0]
    model_fret.update()
    model_fret.find_parameters()
    fit_fret.fit_range = 100, 1500
    
    # Link the lifetime parameter between the two fits to create joint fitting
    print("Linking lifetime parameters...")
    model_fret.parameter_dict['tL1'].link = model_donor.parameter_dict['tL1']
    
    print("Running donor-only fit...")
    chi2_donor_before = fit_donor.chi2
    fit_donor.run()
    chi2_donor_after = fit_donor.chi2
    print(f"Donor fit - Chi² before: {chi2_donor_before:.2f}, after: {chi2_donor_after:.2f}")
    
    print("Running FRET fit...")
    chi2_fret_before = fit_fret.chi2
    fit_fret.run()
    chi2_fret_after = fit_fret.chi2
    print(f"FRET fit - Chi² before: {chi2_fret_before:.2f}, after: {chi2_fret_after:.2f}")
    
    # Print fitted parameters
    print("\nDonor-only fitted parameters:")
    for param_name in model_donor.parameter_names:
        param = model_donor.parameter_dict[param_name]
        print(f"  {param_name}: {param.value:.4f} ± {param.stdev:.4f}")
    
    print("\nFRET fitted parameters:")
    for param_name in model_fret.parameter_names:
        param = model_fret.parameter_dict[param_name]
        linked_info = " (linked)" if param.link is not None else ""
        print(f"  {param_name}: {param.value:.4f} ± {param.stdev:.4f}{linked_info}")
    
    print("Joint fitting completed successfully!")
    
    # Calculate the lifetime ratio as a simple FRET efficiency estimate
    donor_lifetime = model_donor.parameter_dict['tL1'].value
    fret_lifetime = model_fret.parameter_dict['tL1'].value  # Same as donor due to linking
    
    # For a simple estimate, we could compare amplitudes or add a second component
    print(f"\nDonor lifetime: {donor_lifetime:.3f} ns")
    print(f"Note: This simple example links the lifetimes. For actual FRET analysis,")
    print(f"you would need to use the GaussianModel or create a more complex model.")

if __name__ == "__main__":
    main()