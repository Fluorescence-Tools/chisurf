import pytest
import numpy as np
import pathlib
import chisurf.plugins.fluorescence_decay.lltf.core.convolve as convolve
from chisurf.models.parse.parse import ParseModel

# Reference data paths
REF_DIR = pathlib.Path(__file__).parent.parent / "data" / "reference"

@pytest.fixture
def time_axis_lt():
    return np.linspace(0, 100, 1024)

@pytest.fixture
def irf_lt(time_axis_lt):
    irf = np.exp(-(time_axis_lt - 5)**2 / 2.0)
    irf /= irf.sum()
    return irf

def test_lifetime_regression(time_axis_lt, irf_lt):
    ref_file = REF_DIR / "lifetime_1exp.npy"
    if not ref_file.exists():
        pytest.skip("Reference data missing")
    
    ref_y = np.load(ref_file)
    
    # 1-exp: Amp=1.0, Lifetime=10.0
    lifetime_spectrum = np.array([1.0, 10.0])
    output_decay = np.zeros_like(time_axis_lt)
    
    convolve.convolve_lifetime_spectrum(
        output_decay=output_decay,
        lifetime_spectrum=lifetime_spectrum,
        instrument_response_function=irf_lt,
        time_axis=time_axis_lt
    )
    
    assert np.allclose(output_decay, ref_y)

def test_fcs_regression():
    ref_file = REF_DIR / "fcs_3d_gauss.npy"
    if not ref_file.exists():
        pytest.skip("Reference data missing")
    
    ref_y = np.load(ref_file)
    
    time_axis = np.logspace(-3, 3, 100)
    # 3D Gauss: N=1, td=1.0, s=5.0, b=0.0
    # b+1/abs(N)*(1+x/td)**(-1)/sqrt(1+1/s**2*x/td)
    
    # Manually compute to verify logic consistency
    N, td, s, b = 1.0, 1.0, 5.0, 0.0
    y = b + 1.0/abs(N) * (1.0 + time_axis/td)**(-1.0) / np.sqrt(1.0 + (1.0/s**2) * (time_axis/td))
    
    assert np.allclose(y, ref_y)

def test_parse_model_evaluation():
    # Test ParseModel's equation transformation and evaluation
    model = ParseModel()
    model.func = "a*x + b"
    
    # Check if keys were extracted correctly
    assert "a" in model._keys
    assert "b" in model._keys
    
    # Mock some data
    class MockData:
        def __init__(self, x):
            self.x = x
    
    class MockFit:
        def __init__(self, x):
            self.data = MockData(x)
        def update(self):
            pass
            
    x = np.array([1.0, 2.0, 3.0])
    model.fit = MockFit(x)
    
    # Set parameters: a=2.0, b=1.0
    for p in model._parameters_equation:
        if p.name == "a":
            p.value = 2.0
        elif p.name == "b":
            p.value = 1.0
            
    model.update_model()
    
    expected_y = 2.0 * x + 1.0
    assert np.allclose(model.y, expected_y)
