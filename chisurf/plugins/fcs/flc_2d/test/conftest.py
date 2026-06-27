"""Shared fixtures for the 2D-FLC validation suite.

The tests are driven by the reference data shipped with the original MATLAB code
(``thirdparty/2D-FLC-code/``): a simulated single-molecule photon stream with two
fluorescence-lifetime species (tau = 1 and 3 ns), equal brightness, interconversion rate
matrix ``[[0,30],[10,0]]`` s^-1 (relaxation 40 s^-1 => 25 ms), equilibrium populations
0.75 / 0.25, and the instrument response function. If that data is not present the
data-driven tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _find_reference_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "thirdparty" / "2D-FLC-code"
        if (cand / "simulated_data.mat").exists():
            return cand
    return None


REF_DIR = _find_reference_dir()
GROUND_TRUTH = {
    "lifetimes_ns": (1.0, 3.0),
    "relaxation_rate_per_s": 40.0,
    "relaxation_time_s": 0.025,
    "eq_populations": (0.75, 0.25),
    "micro_time_resolution_ns": 0.004,
}


@pytest.fixture(scope="session")
def reference_mat():
    """Load the simulated reference data set (skips if unavailable)."""
    if REF_DIR is None:
        pytest.skip("reference 2D-FLC-code/simulated_data.mat not available")
    import scipy.io as sio

    m = sio.loadmat(str(REF_DIR / "simulated_data.mat"), squeeze_me=True, struct_as_record=False)
    return {
        "macro_seconds": np.asarray(m["tt1"], dtype=float),
        "micro_ticks": np.asarray(m["kin1"], dtype=np.int64),
        "micro_resolution_ns": float(m["tstep"]),
        "irf": np.asarray(m["IRF"], dtype=float),
        "irf_time_ns": np.asarray(m["xdata"], dtype=float),
        "n_microtime_bins": 3127,
    }


@pytest.fixture(scope="session")
def reference_photons(reference_mat):
    """Macro times as ascending integer 1-us ticks plus aligned micro ticks."""
    macro_res_s = 1e-6
    macro = np.rint(reference_mat["macro_seconds"] / macro_res_s).astype(np.int64)
    order = np.argsort(macro, kind="stable")
    return {
        "macro_ticks": macro[order],
        "micro_ticks": reference_mat["micro_ticks"][order],
        "macro_resolution_s": macro_res_s,
        "micro_resolution_ns": reference_mat["micro_resolution_ns"],
        "irf": reference_mat["irf"],
        "irf_time_ns": reference_mat["irf_time_ns"],
        "n_microtime_bins": reference_mat["n_microtime_bins"],
    }
