#!/usr/bin/env python3

"""
Script to perform joint fitting of TCSPC data from the ibh_sample folder.
This demonstrates how to load multiple TCSPC datasets and fit them jointly.
"""

import pathlib
import sys

# Add the chisurf package to the path
chisurf_path = pathlib.Path(__file__).parent
sys.path.insert(0, str(chisurf_path))

import chisurf.experiments
import chisurf.models
import chisurf.fitting
import chisurf.fluorescence.tcspc

def main():
    print("Starting joint TCSPC fitting...")
    
    # Set up the experiment and reader
    dt = 0.0141  # Time per channel in ns (8 ps per channel based on file names)
    tcspc_experiment = chisurf.experiments.core.Experiment(name='TCSPC Joint Fit')
    tcspc_reader = chisurf.experiments.tcspc.TCSPCReader(
        is_jordi=False,
        skiprows=10,
        dt=dt,
        experiment=tcspc_experiment
    )
    
    # Load the data files
    base_path = pathlib.Path('./test/data/tcspc/ibh_sample')
    
    # Load IRF (Instrument Response Function)
    print("Loading IRF...")
    irf = tcspc_reader.read(filename=str(base_path / 'Prompt.txt'))
    
    # Load decay data
    print("Loading decay data...")
    decay_donor_only = tcspc_reader.read(filename=str(base_path / 'Decay_577D.txt'))
    decay_fret = tcspc_reader.read(filename=str(base_path / 'Decay_577D+577A+GTPgS.txt'))
    
    # Create a data group containing both datasets for joint fitting
    print("Creating data group...")
    data_group = chisurf.data.DataGroup([decay_donor_only[0], decay_fret[0]])
    
    # Create individual fits first, then we'll link them
    print("Creating individual fits...")
    
    # Fit for donor-only decay
    fit_donor = chisurf.fitting.fit.FitGroup(
        data=chisurf.data.DataGroup([decay_donor_only[0]]),
        model_class=chisurf.models.tcspc.lifetime.LifetimeModel
    )
    
    # Fit for FRET decay
    fit_fret = chisurf.fitting.fit.FitGroup(
        data=chisurf.data.DataGroup([decay_fret[0]]),
        model_class=chisurf.models.tcspc.fret.GaussianModel
    )
    
    # Configure the donor-only model
    print("Configuring donor-only model...")
    donor_model = fit_donor.model
    donor_model.lifetimes.append()  # Add one lifetime component
    donor_model.convolve._irf = irf[0]
    donor_model.update()
    donor_model.find_parameters()
    fit_donor.fit_range = 100, 1500
    
    # Configure the FRET model
    print("Configuring FRET model...")
    fret_model = fit_fret.model
    
    # Add a Gaussian distance distribution for the FRET species
    fret_model.append(
        mean=50,      # Mean distance in Å
        sigma=6,     # Width of distribution
        species_fraction=0.5  # Fraction of molecules showing FRET
    )
    
    # Set the IRF for convolution
    fret_model.convolve._irf = irf[0]
    
    # Link the donor lifetime from the donor-only fit to the FRET fit
    # This creates a joint fitting scenario where the donor lifetime is shared
    fret_model.parameter_dict['tL1'].link = donor_model.parameter_dict['tL1']
    
    # Find initial parameters
    fret_model.find_parameters()
    fit_fret.fit_range = 100, 1500
    
    print("Model parameters:")
    for param_name in model.parameter_names:
        param = model.parameter_dict[param_name]
        print(f"  {param_name}: {param.value:.3f} (fixed: {param.fixed})")
    
    # Run the fits (they are now linked through the shared donor lifetime)
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
    
    # Print fitted parameters from both fits
    print("\nDonor-only fitted parameters:")
    for param_name in donor_model.parameter_names:
        param = donor_model.parameter_dict[param_name]
        print(f"  {param_name}: {param.value:.4f} ± {param.stdev:.4f}")
    
    print("\nFRET fitted parameters:")
    for param_name in fret_model.parameter_names:
        param = fret_model.parameter_dict[param_name]
        linked_info = " (linked)" if param.link is not None else ""
        print(f"  {param_name}: {param.value:.4f} ± {param.stdev:.4f}{linked_info}")
    
    print("Joint fitting completed successfully!")
    
    # Save the fit results
    save_path = pathlib.Path('./joint_fit_results')
    save_path.mkdir(exist_ok=True)
    chisurf.fitting.fit.save_fit(str(save_path / 'joint_fit.chisurf'), joint_fit)
    print(f"Fit results saved to: {save_path}")

if __name__ == "__main__":
    main()