"""Unit tests for lightpath simulator physics logic."""

import numpy as np
import pytest
from chisurf.plugins._dev.lightpath_simulator.crosstalk import calculate_r0, WAVELENGTHS


def test_calculate_r0_basic():
    """Test R0 calculation for overlapping box spectra."""
    # Create simple overlapping box spectra
    donor_em = np.zeros_like(WAVELENGTHS)
    donor_em[(WAVELENGTHS >= 480) & (WAVELENGTHS <= 520)] = 1.0
    
    acceptor_abs = np.zeros_like(WAVELENGTHS)
    acceptor_abs[(WAVELENGTHS >= 480) & (WAVELENGTHS <= 520)] = 1.0
    
    # Using typical values: QY=1, EC=100k
    r0 = calculate_r0(
        donor_em=donor_em,
        donor_qy=1.0,
        acceptor_abs=acceptor_abs,
        acceptor_ec_max=100000.0,
        kappa2=2/3,
        n=1.33
    )
    
    assert r0 > 0
    # For perfect overlap, R0 should be substantial
    assert 40 < r0 < 100


def test_calculate_r0_no_overlap():
    """Test R0 is zero when spectra do not overlap."""
    donor_em = np.zeros_like(WAVELENGTHS)
    donor_em[(WAVELENGTHS >= 350) & (WAVELENGTHS <= 400)] = 1.0
    
    acceptor_abs = np.zeros_like(WAVELENGTHS)
    acceptor_abs[(WAVELENGTHS >= 500) & (WAVELENGTHS <= 550)] = 1.0
    
    r0 = calculate_r0(
        donor_em=donor_em,
        donor_qy=1.0,
        acceptor_abs=acceptor_abs,
        acceptor_ec_max=100000.0,
        kappa2=2/3,
        n=1.33
    )
    
    assert r0 == 0.0


def test_calculate_r0_scaling():
    """Test that R0 scales correctly with QY and EC."""
    donor_em = np.zeros_like(WAVELENGTHS)
    donor_em[(WAVELENGTHS >= 480) & (WAVELENGTHS <= 520)] = 1.0
    
    acceptor_abs = np.zeros_like(WAVELENGTHS)
    acceptor_abs[(WAVELENGTHS >= 480) & (WAVELENGTHS <= 520)] = 1.0
    
    r0_ref = calculate_r0(donor_em, 1.0, acceptor_abs, 100000.0)
    
    # If QY is halved, R0 should decrease by factor of (1/2)^(1/6) approx 0.89
    r0_half_qy = calculate_r0(donor_em, 0.5, acceptor_abs, 100000.0)
    assert r0_half_qy < r0_ref
    assert pytest.approx(r0_half_qy / r0_ref, rel=1e-3) == (0.5)**(1/6)
    
    # Same for EC
    r0_half_ec = calculate_r0(donor_em, 1.0, acceptor_abs, 50000.0)
    assert pytest.approx(r0_half_ec / r0_ref, rel=1e-3) == (0.5)**(1/6)
