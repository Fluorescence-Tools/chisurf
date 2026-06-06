import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "chisurf"
    / "plugins"
    / "jordi_g_factor"
    / "__init__.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "chisurf_plugins_jordi_g_factor", _MODULE_PATH
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = _MOD
_SPEC.loader.exec_module(_MOD)

JordiGFactorCalculator = _MOD.JordiGFactorCalculator


def test_solve_linked_l_from_steady_state_recovers_reference() -> None:
    sp = 1000.0
    ss = 620.0
    g = 1.17
    l_ref = 0.034

    num = g * sp - ss
    den = (1.0 - 3.0 * l_ref) * g * sp + (2.0 - 3.0 * l_ref) * ss
    r_target = num / den

    l_est = JordiGFactorCalculator._solve_linked_l_from_steady_state(
        sp=sp,
        ss=ss,
        g_factor=g,
        r_target=r_target,
    )
    assert l_est == pytest.approx(l_ref, rel=1e-12, abs=1e-12)


def test_estimate_lifetime_first_moment_returns_weighted_mean() -> None:
    t = [0.0, 1.0, 2.0, 3.0]
    i = [0.0, 2.0, 2.0, 0.0]

    tau = JordiGFactorCalculator._estimate_lifetime_first_moment(t, i)

    assert tau == pytest.approx(0.5, rel=1e-12, abs=1e-12)
