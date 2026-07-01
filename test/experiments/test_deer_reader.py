"""Round-trip tests for the DEER experiment reader (CSV + real Bruker data)."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from chisurf.core.experiments.deer import DeerReader
from chisurf.core.experiments.deer.reader import phase_correct

#: Real experimental DEER sample files bundled under ``test/data/deer``.
_DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "deer"


def _write_csv(tmp_path, with_imag=False):
    t = np.linspace(0.0, 3.0, 128)
    v = 0.6 + 0.4 * np.cos(2 * np.pi * t / 0.8) * np.exp(-t)
    path = tmp_path / "trace.csv"
    with open(path, "w") as fh:
        fh.write("t[us],Vreal" + (",Vimag" if with_imag else "") + "\n")
        for i in range(t.size):
            row = f"{t[i]:.6f},{v[i]:.6f}"
            if with_imag:
                row += ",0.0"
            fh.write(row + "\n")
    return path, t, v


def test_reader_loads_csv(tmp_path):
    path, t, v = _write_csv(tmp_path)
    reader = DeerReader()
    group = reader.read(filename=str(path))
    assert len(group) == 1
    data = group[0]
    assert data.y.size == t.size
    assert "deer" in data.meta_data
    meta = data.meta_data["deer"]
    for key in ("t", "V", "V_imag", "t0", "exp_type", "scale"):
        assert key in meta
    # Normalised to V(t0) = 1.
    assert abs(float(np.max(data.y)) - 1.0) < 1e-6
    assert np.all(np.isfinite(data.y))


def test_reader_autofitrange(tmp_path):
    path, t, _ = _write_csv(tmp_path)
    reader = DeerReader()
    group = reader.read(filename=str(path))
    assert reader.autofitrange(group[0]) == (0, t.size)


def test_reader_missing_file_returns_empty_group():
    reader = DeerReader()
    group = reader.read(filename="/nonexistent/does_not_exist.csv")
    assert len(group) == 0


def test_phase_correction_rotates_into_real_channel():
    t = np.linspace(0.0, 3.0, 100)
    signal = (0.5 + 0.5 * np.cos(2 * np.pi * t)) * np.exp(-0.3 * t)
    phi = 0.7
    v = signal * np.exp(1j * phi)  # spill amplitude into the imaginary channel
    real, imag, phase = phase_correct(v)
    assert np.allclose(real, signal, atol=1e-6)
    assert np.max(np.abs(imag)) < 1e-6


# --- real experimental data (bundled Bruker BES3T samples in test/data/deer) --
@pytest.mark.parametrize(
    "stem, t_end_us",
    [("deer_ringtest_4pdeer", 2.83), ("deer_twostate", 4.70)],
)
def test_reader_loads_real_bruker_bes3t(stem, t_end_us):
    dsc = _DATA / f"{stem}.DSC"
    if not dsc.is_file():
        pytest.skip(f"missing test data {dsc}")
    group = DeerReader().read(filename=str(dsc))
    assert len(group) == 1
    data = group[0]
    meta = data.meta_data["deer"]
    # Time axis in microseconds, phase-corrected & normalised to V(t0)=1.
    assert data.y.size > 100
    assert abs(float(data.x[-1]) - t_end_us) < 0.1
    assert abs(float(np.max(data.y)) - 1.0) < 1e-6
    assert np.all(np.isfinite(data.y))
    assert np.any(meta["V_imag"])  # complex trace -> non-zero imaginary channel


def test_reader_loads_real_csv():
    csv = _DATA / "deer_trace.csv"
    if not csv.is_file():
        pytest.skip(f"missing test data {csv}")
    group = DeerReader().read(filename=str(csv))
    data = group[0]
    # Headerless ns axis is auto-detected -> a few-µs DEER window, not hundreds.
    assert 1.0 < float(data.x[-1]) < 20.0
    assert np.all(np.isfinite(data.y))


def test_real_bruker_fits_gaussian():
    """A single-Gaussian DEER fit on real ring-test data converges finitely."""
    import chisurf.core.fitting.fit as fit_mod
    from chisurf.core.models.deer.deer import DeerGaussianModel

    dsc = _DATA / "deer_ringtest_4pdeer.DSC"
    if not dsc.is_file():
        pytest.skip(f"missing test data {dsc}")
    group = DeerReader().read(filename=str(dsc))
    fit = fit_mod.Fit(model_class=DeerGaussianModel, data=group[0])
    fit.xmin, fit.xmax = 0, len(fit.data.y)
    fit.run()
    model = fit.model
    r_fit = model.gaussians.means[0]
    assert 15.0 < r_fit < 100.0          # physically plausible spin-label distance (Å)
    assert np.all(np.isfinite(model.y))
    assert np.isfinite(fit.chi2r)
