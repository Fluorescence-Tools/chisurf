#!/usr/bin/env python3

"""
Direct test of GaussianModel creation to understand the 'donor' attribute issue.
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
    print("Testing GaussianModel creation directly...")
    
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
    
    print("Loading decay data...")
    decay_dd_da = tcspc_reader.read(filename=str(base_path / 'Decay_577D+577A+GTPgS.txt'))
    
    print("Creating fit with GaussianModel...")
    try:
        fit_da = chisurf.core.fitting.fit.FitGroup(
            data=decay_dd_da,
            model_class=chisurf.core.models.tcspc.fret.GaussianModel
        )
        print("Fit created successfully!")
        
        model_da = fit_da.model
        print(f"Model type: {type(model_da)}")
        print(f"Model attributes: {dir(model_da)}")
        
        # Check if donor attribute exists
        if hasattr(model_da, 'donor'):
            print(f"Donor attribute exists: {model_da.donor}")
        else:
            print("Donor attribute does not exist")
            
    except Exception as e:
        print(f"Error creating fit: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()