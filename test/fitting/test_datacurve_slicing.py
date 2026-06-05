
import numpy as np
import chisurf.core.data
import chisurf.core.fitting
import chisurf.core.curve
import tempfile
import os

def test_datacurve_slicing_content():
    """Verify that DataCurve slicing returns correct (x, y) data, not flipped axes."""
    x = np.arange(10, 20, dtype=float) # Lag times 10..19
    y = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]) # Amplitudes
    ex = np.zeros_like(x)
    ey = np.ones_like(y) * 0.1
    mask = np.ones_like(y)
    
    dc = chisurf.core.data.DataCurve(x=x, y=y, ex=ex, ey=ey, mask=mask)
    
    # Check __getitem__ content
    # Regression check: Curve.__getitem__ used to return (0..N-1, x) instead of (x, y)
    x_slice, y_slice, ex_slice, ey_slice, mask_slice = dc[0:5]
    
    assert np.array_equal(x_slice, x[0:5]), f"X error: expected {x[0:5]}, got {x_slice}"
    assert np.array_equal(y_slice, y[0:5]), f"Y error: expected {y[0:5]}, got {y_slice}"
    assert np.array_equal(ey_slice, ey[0:5]), f"EY error: expected {ey[0:5]}, got {ey_slice}"

def test_calculate_weighted_residuals_with_datacurve():
    """Verify calculate_weighted_residuals works with DataCurve output."""
    x = np.arange(10, dtype=float)
    y = np.ones(10)
    ey = np.ones(10) * 0.1
    dc = chisurf.core.data.DataCurve(x=x, y=y, ex=np.zeros(10), ey=ey)
    
    model = chisurf.core.curve.Curve(x=x, y=y - 0.1)
    
    # (y - (y-0.1)) / 0.1 = 0.1 / 0.1 = 1.0
    wres = chisurf.core.fitting.calculate_weighted_residuals(dc, model, 0, 5)
    expected_wres = np.ones(5)
    assert np.allclose(wres, expected_wres)

if __name__ == "__main__":
    test_datacurve_slicing_content()
    test_calculate_weighted_residuals_with_datacurve()
    print("DataCurve slicing and residuals tests passed!")
