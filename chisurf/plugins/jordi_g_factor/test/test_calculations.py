import numpy as np
from chisurf.plugins.jordi_g_factor.core.calculations import (
    calculate_g_factor_core,
    perrin_steady_state_anisotropy,
    solve_linked_l_from_steady_state
)


def test_calculate_g_factor_core():
    # Make a clean decay with known g-factor ratio = 1.5
    t = np.linspace(0, 10, 1000)
    decay = np.exp(-t / 2.0)
    par = decay
    perp = decay / 1.5

    # Tail bounds in channel units (0 to 1000)
    res = calculate_g_factor_core(
        parallel_data=par,
        perpendicular_data=perp,
        region_bounds=[700, 900],
        decay_shift=0.0,
        use_bg=False,
    )
    assert res["g_factor"] is not None
    assert np.allclose(res["g_factor"], 1.5, atol=1e-3)


def test_perrin_steady_state():
    val = perrin_steady_state_anisotropy(tau_ns=4.0, rho_ns=16.0, r0=0.38)
    assert np.allclose(val, 0.38 / (1.0 + 4.0 / 16.0))


def test_solve_linked_l():
    sp = 1000.0
    ss = 600.0
    g = 1.5
    # g*sp - ss = 1.5*1000 - 600 = 900
    # g*sp + 2*ss = 1500 + 1200 = 2700
    # g*sp + ss = 2100
    # target r = 0.3
    # 3*l = (2700 - 900/0.3) / 2100 = (2700 - 3000) / 2100 = -300 / 2100 = -1/7
    # l = -1/21
    val = solve_linked_l_from_steady_state(sp=sp, ss=ss, g_factor=g, r_target=0.3)
    assert np.allclose(val, -1.0 / 21.0)
