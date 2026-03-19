import importlib.util
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "chisurf" / "fluorescence" / "anisotropy" / "integrals.py"
_SPEC = importlib.util.spec_from_file_location("chisurf_fluorescence_anisotropy_integrals", _MODULE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

anisotropy_from_integrals = _MOD.anisotropy_from_integrals
compute_g_factor_perrin = _MOD.compute_g_factor_perrin
perrin_steady_state_anisotropy = _MOD.perrin_steady_state_anisotropy


def test_perrin_steady_state_anisotropy_monotonic() -> None:
    r_fast = perrin_steady_state_anisotropy(tau=4.0, rho=0.5, r0=0.38)
    r_slow = perrin_steady_state_anisotropy(tau=4.0, rho=20.0, r0=0.38)
    assert 0.0 <= r_fast < r_slow < 0.38


def test_compute_g_factor_perrin_matches_equation_22422() -> None:
    sp = [1000.0, 950.0, 900.0]
    ss = [650.0, 610.0, 580.0]
    l1 = 0.03
    l2 = 0.02
    g = compute_g_factor_perrin(sp, ss, tau=3.2, rho=6.8, r0=0.38, l1=l1, l2=l2)
    res = anisotropy_from_integrals(sp, ss, G=g, l1=l1, l2=l2)
    r_target = perrin_steady_state_anisotropy(tau=3.2, rho=6.8, r0=0.38)
    assert res.r_e == pytest.approx(r_target, rel=1e-10, abs=1e-12)


def test_anisotropy_from_integrals_scatter_branch() -> None:
    res = anisotropy_from_integrals(
        s_p=[100.0, 120.0, 110.0],
        s_s=[60.0, 58.0, 62.0],
        G=1.1,
        l1=0.02,
        l2=0.01,
        gamma=0.1,
        B_p=10.0,
        B_s=8.0,
        scatter_corrected=True,
    )
    assert isinstance(res.r_e, float)
    assert isinstance(res.r_s, float)
    assert isinstance(res.chi, float)
    assert isinstance(res.S_ges, float)
