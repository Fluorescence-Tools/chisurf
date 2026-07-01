"""Cross-check ChiSurf RICS against PAM (Mia) on RICS_EGFPGFP.

PAM's Mia module computed the RICS auto-correlation of ``RICS_EGFPGFP.tif`` and
stored it as a ``.miacor`` MATLAB file. Acquisition parameters (from the PAM
``_Info.txt``): pixel time 11.1111 µs, line time 3.3333 ms, pixel size 40 nm,
ROI 1,1,200,200, 50 frames, "Frame mean" correction.

This test reads the same TIFF (validating the LZW multi-frame reader),
computes ChiSurf's RICS correlation on the identical 200x200 ROI, and asserts
the baseline-subtracted correlation *decay* matches PAM's within tolerance.
The correlation function is the quantity PAM stores (a fitted D is derived from
it), so agreement here means both packages recover the same diffusion.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

_DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "rics"
_TIF = _DATA / "RICS_EGFPGFP.tif"
_PAM = _DATA / "RICS_EGFPGFP_PAM.miacor"

pytestmark = pytest.mark.skipif(
    not (_TIF.exists() and _PAM.exists()),
    reason="RICS_EGFPGFP test data not present",
)


def _pam_correlation() -> np.ndarray:
    """Return PAM's 200x200 RICS correlation (peak-centred)."""
    import scipy.io as sio

    m = sio.loadmat(str(_PAM))
    return np.asarray(m["Data"][0, 0], dtype=float)


def _decay_profiles(corr: np.ndarray, k: int = 15):
    """Return baseline-subtracted, peak-normalised fast/slow decay profiles.

    Lag 0 (the shot-noise spike) is excluded; the baseline is the median of the
    image border.
    """
    pk = np.unravel_index(np.argmax(corr), corr.shape)
    border = np.concatenate([corr[0, :], corr[-1, :], corr[:, 0], corr[:, -1]])
    g = corr - np.median(border)
    peak = g[pk]
    fast = g[pk[0], pk[1]: pk[1] + k + 1] / peak
    slow = g[pk[0]: pk[0] + k + 1, pk[1]] / peak
    return fast[1:], slow[1:]  # drop lag 0


def test_rics_reader_reads_lzw_stack():
    """The RICS reader loads the LZW-compressed multi-frame TIFF as a stack."""
    from chisurf.core.experiments.rics import RICSReader

    g = RICSReader(name="RICS", reading_routine="PTU", channel=0).read(filename=str(_TIF))
    assert g is not None and len(g) == 1
    meta = g[0].meta_data["rics"]
    assert np.asarray(meta["ics_mean"]).shape == (300, 300)
    assert int(meta.get("n_frames", 0)) == 50


def test_rics_correlation_matches_pam():
    """ChiSurf's RICS correlation decay matches PAM's on the same 200x200 ROI."""
    import tifffile

    from chisurf.core.experiments.rics.ics_core import compute_rics_from_images

    stack = np.asarray(tifffile.imread(str(_TIF)), dtype=float)
    cs = np.asarray(
        compute_rics_from_images(stack[:, :200, :200], use_fftshift=True).ics_mean,
        dtype=float,
    )
    pam = _pam_correlation()
    assert cs.shape == pam.shape == (200, 200)

    cs_fast, cs_slow = _decay_profiles(cs)
    pam_fast, pam_slow = _decay_profiles(pam)

    # The baseline-subtracted decays are highly correlated in shape.
    r_fast = np.corrcoef(cs_fast, pam_fast)[0, 1]
    r_slow = np.corrcoef(cs_slow, pam_slow)[0, 1]
    assert r_fast > 0.95, f"fast-axis RICS decay vs PAM only r={r_fast:.3f}"
    assert r_slow > 0.95, f"slow-axis RICS decay vs PAM only r={r_slow:.3f}"

    # Amplitudes agree to better than 0.06 (absolute, peak-normalised).
    assert np.max(np.abs(cs_fast - pam_fast)) < 0.06
    assert np.max(np.abs(cs_slow - pam_slow)) < 0.06
