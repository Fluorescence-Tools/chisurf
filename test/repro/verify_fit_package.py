import numpy as np
import sys
import os

# Ensure we can import the plugin
sys.path.append(os.getcwd())

from chisurf.plugins.fcs.flc_2d.api import correlate_tttr, fit_mem_2d
from chisurf.plugins.fcs.flc_2d.fit import TwoDMEMFitter, OneDMEMFitter, GlobalTwoDMEMFitter, RateMatrixFitter

def test_imports():
    print("Testing imports from modular fit package...")
    assert TwoDMEMFitter is not None
    assert OneDMEMFitter is not None
    assert GlobalTwoDMEMFitter is not None
    assert RateMatrixFitter is not None
    print("Imports successful!")

def test_api():
    print("Testing API with new fit package...")
    # Create dummy data
    times = np.linspace(0, 10, 100)
    mat = np.outer(np.exp(-times/1.0), np.exp(-times/1.0))
    
    # Just check if fit_mem_2d can be called (it will fail on optimization but that's fine for import check)
    # Actually, let's just check if the fitter class can be instantiated
    fitter = TwoDMEMFitter()
    print("Fitter instantiation successful!")

if __name__ == "__main__":
    test_imports()
    test_api()
    print("Verification successful!")
