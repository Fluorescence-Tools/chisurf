from __future__ import annotations

"""DEER/PELDOR experiment reader.

Loads a dipolar time-domain trace ``V(t)`` from Bruker BES3T (``.DSC``/``.DTA``)
or CSV/text files and wraps it into a
:class:`chisurf.core.data.DataCurve` with a ``meta_data['deer']`` payload that
the DEER models consume. Self-contained (numpy/scipy only).
"""

import pathlib

import numpy as np

import chisurf.core.data
from chisurf.core.experiments.core.reader import ExperimentReader

from .csv_loader import load_csv
from .eprload import find_companion, load_bes3t


def estimate_noise_level(v_real: np.ndarray, v_imag: np.ndarray | None = None) -> float:
    """Estimate the noise standard deviation of a DEER trace.

    Two robust estimators, in order of preference:

    1. the (median-absolute-deviation) spread of the imaginary channel, which
       for a well-phased DEER signal contains essentially pure noise;
    2. a second-derivative estimate on the real channel — for white noise
       ``var(diff(V, 2)) = 6 * sigma**2`` — made robust with the MAD.

    The value is used as the per-point weight so that a good fit gives
    ``chi2_red ~ 1`` for a good fit (normalising by the estimated noise level).
    """
    def _mad_sigma(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(1.4826 * np.median(np.abs(x - np.median(x))))

    if v_imag is not None and np.any(v_imag):
        s_im = _mad_sigma(v_imag)
        if s_im > 1e-9:
            return s_im
    d2 = np.diff(np.asarray(v_real, dtype=float), n=2)
    if d2.size == 0:
        return 1e-6
    return max(_mad_sigma(d2) / np.sqrt(6.0), 1e-9)


def phase_correct(v_complex: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate a complex trace to put the signal into the real channel.

    Returns ``(real, imag, phase)`` where ``phase`` is the applied rotation
    (radians). Uses the zero-order phase that maximises the real-part sum.
    """
    v = np.asarray(v_complex, dtype=complex)
    phase = float(np.angle(np.sum(v))) if np.any(v.imag) else 0.0
    rot = v * np.exp(-1j * phase)
    # Prefer the sign that makes the leading amplitude positive.
    if rot.real[:max(1, len(rot) // 20)].mean() < 0:
        rot = -rot
        phase += np.pi
    return rot.real, rot.imag, phase


class DeerReader(ExperimentReader):
    """Reader for 4-pulse DEER time-domain traces."""

    name: str = "DEER (BES3T/CSV)"

    def __init__(
        self,
        name: str = "DEER (BES3T/CSV)",
        phase_correction: bool = True,
        normalize: bool = True,
        exp_type: str = "4pDEER",
        *args,
        **kwargs,
    ):
        """Initialize a DEER reader.

        Parameters
        ----------
        name : str
            Human-readable reader name.
        phase_correction : bool
            Whether to zero-order phase-correct complex traces.
        normalize : bool
            Whether to normalise the trace to unity at the zero time.
        exp_type : str
            DEER experiment label stored in the metadata (informational).
        *args, **kwargs
            Forwarded to :class:`ExperimentReader`.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.phase_correction = bool(phase_correction)
        self.normalize = bool(normalize)
        self.exp_type = str(exp_type)

    def autofitrange(self, data, **kwargs):
        """Return the full data range as the default fit interval."""
        try:
            return 0, len(data.y)
        except Exception:
            return 0, 0

    # -- loading --
    def _load_arrays(self, fn: pathlib.Path):
        """Return ``(t_us, v_real, v_imag)`` for a supported DEER file."""
        suffix = fn.suffix.lower()
        if suffix in (".dsc", ".dta"):
            dsc, dta = find_companion(str(fn))
            t, v, _attrs = load_bes3t(dsc, dta)
            v = np.asarray(v)
            if np.iscomplexobj(v):
                return t, v.real, v.imag
            return t, v.astype(float), np.zeros_like(t)
        # CSV / text / ASCII
        t, vr, vi, _attrs = load_csv(str(fn))
        return t, vr, vi

    def read(self, filename: str = None, *args, **kwargs):
        """Read a DEER file and return an :class:`ExperimentDataCurveGroup`."""
        group = chisurf.core.data.ExperimentDataCurveGroup([])
        if filename is None:
            return group
        if isinstance(filename, (list, tuple)):
            if not filename:
                return group
            filename = filename[0]
        fn = pathlib.Path(filename)
        if not fn.is_file():
            return group

        t, v_real, v_imag = self._load_arrays(fn)

        # Phase correction (complex data only).
        if self.phase_correction and np.any(v_imag):
            v_real, v_imag, phase = phase_correct(v_real + 1j * v_imag)
        else:
            phase = 0.0

        # Zero time = position of maximum amplitude (near the start for DEER).
        t0_idx = int(np.argmax(v_real))
        t0 = float(t[t0_idx])

        # Normalisation to V(t0) = 1.
        scale = float(v_real[t0_idx]) if self.normalize and v_real[t0_idx] != 0 else 1.0
        v_norm = v_real / scale
        v_imag_norm = v_imag / abs(scale) if scale else v_imag

        # Proper per-point noise level so weighted residuals give chi2_red ~ 1.
        sigma = estimate_noise_level(v_norm, v_imag_norm)
        ey = np.full_like(v_norm, sigma)

        deer_meta = {
            "t": np.asarray(t, dtype=float),
            "V": np.asarray(v_norm, dtype=float),
            "V_imag": np.asarray(v_imag, dtype=float),
            "phase": phase,
            "t0": t0,
            "exp_type": self.exp_type,
            "scale": scale,
            "noise_level": float(sigma),
        }

        data = chisurf.core.data.DataCurve(
            name=fn.stem,
            x=np.asarray(t, dtype=float),
            y=np.asarray(v_norm, dtype=float),
            ey=ey,
            filename=str(fn),
            data_reader=self,
            meta_data={"deer": deer_meta},
            load_filename_on_init=False,
        )
        group.append(data)
        group.data_reader = self
        return group
