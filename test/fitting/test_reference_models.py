print("DEBUG: SCRIPT LOADED")

import sys
import pathlib
# Use absolute path of the repository root
TOPDIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(TOPDIR))

import utils
import os
import pytest
import numpy as np

utils.set_search_paths(TOPDIR)

import chisurf.data
from chisurf.fitting.fit import Fit
import chisurf.models.tcspc.lifetime
import chisurf.models.tcspc.nusiance

def generate_synthetic_decay(lifetime, amplitude, background, dt, n_channels):
    time = np.arange(n_channels).astype(np.float32) * dt
    y = amplitude * np.exp(-time / lifetime) + background
    # Add some Poisson noise
    y = np.random.poisson(y).astype(np.float32)
    return time, y

def test_lifetime_model_convergence():
    print("DEBUG: test_lifetime_model_convergence START")
    # 1. Setup Ground Truth
    true_tau = 4.0
    true_amp = 1000.0
    true_bg = 10.0
    dt = 0.032
    n_channels = 1024
    
    time, y_data = generate_synthetic_decay(true_tau, true_amp, true_bg, dt, n_channels)
    ey = np.sqrt(np.maximum(y_data, 1.0)) # Poisson errors
    
    data = chisurf.data.DataCurve(x=time, y=y_data, ey=ey)
    dg = chisurf.data.DataGroup([data])
    
    fit = Fit(model_class=chisurf.models.tcspc.lifetime.LifetimeModel)
    model = fit.model
    # Manual data setup
    model.data.time = time
    model.data.counts = y_data
    model.convolve.do_convolution = False  # Matches synthetic generation
    
    # 3. Add Component and Initial Guesses
    # amplitudes and lifetimes are Port objects or similar
    model.lifetime_spectrum = np.array([800.0, 3.8], dtype=np.float32)
    model.generic.background = 8.0
    
    fit.run()
    
    # 4. Assertions
    # retrieve current values
    spectrum = model.lifetime_spectrum
    fitted_amp = spectrum[0]
    fitted_tau = spectrum[1]
    fitted_bg = model.generic.background
    
    print(f"True Tau: {true_tau}, Fitted Tau: {fitted_tau}")
    print(f"True Amp: {true_amp}, Fitted Amp: {fitted_amp}")
    print(f"True BG: {true_bg}, Fitted BG: {fitted_bg}")
    
    assert np.isclose(fitted_tau, true_tau, rtol=0.1)
    assert np.isclose(fitted_amp, true_amp, rtol=0.2)
    assert np.isclose(fitted_bg, true_bg, rtol=0.5)
    assert fit.chi2r < 1.5

if __name__ == "__main__":
    test_lifetime_model_convergence()
    print("Test passed!")
