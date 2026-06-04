from __future__ import annotations

import pathlib
import warnings

import numpy as np
import scipy.io

import chisurf
import chisurf.fluorescence.fcs

from chisurf import typing
from chisurf.fio.fluorescence.fcs.definitions import FCSDataset


def _load_nested_mat(path: pathlib.Path) -> dict:
    """Load a MATLAB .mat file and convert nested ``mat_struct`` objects.

    The Jonas Ries SFCS tools store their data in a nested ``g`` structure.
    SciPy exposes this as ``mat_struct`` instances; here we turn them into
    plain Python dictionaries so that field access is straightforward.
    """

    # Suppress architecture-related scipy.io warnings; this mirrors what
    # the original SFCS tools do while keeping the behavior explicit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = scipy.io.loadmat(
            str(path), struct_as_record=False, squeeze_me=True
        )

    try:
        # Import here to avoid hard dependence at module import time if
        # SciPy is missing; in that case, loadmat above will already have
        # failed with an informative error.
        from scipy.io.matlab.mio5_params import mat_struct  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - very unlikely when scipy.io is present
        mat_struct = ()  # type: ignore[assignment]

    def _convert(obj):  # type: ignore[override]
        """Recursively convert mat_struct objects to plain dicts.

        Parameters
        ----------
        obj : any
            Object to convert.

        Returns
        -------
        any
            Converted plain Python object.
        """
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, mat_struct):  # type: ignore[arg-type]
            out: dict[str, typing.Any] = {}
            for name in getattr(obj, "_fieldnames", []) or []:
                out[name] = _convert(getattr(obj, name))
            return out
        return obj

    return _convert(data)


def _build_correlation_entries(g_struct: dict, filename: str) -> tuple[list[np.ndarray], list[np.ndarray | None], list[str]]:
    """Return (correlations, traces, labels) from a Ries-style ``g`` dict.

    This follows the structure used by the SFCS.m tools:

    - ``g['ac']`` / ``g['act']`` for autocorrelations and lag times
    - ``g['dc']`` / ``g['dct']`` for dual-color correlations
    - ``g['twof']`` / ``g['twoft']`` for two-focus correlations
    - ``g['dc2f']`` / ``g['dc2ft']`` for dual-color two-focus correlations

    Intensity information is typically provided only as trace averages in
    ``g['trace']``; we convert those into minimal two-point traces so that
    downstream tools can still estimate mean countrate and acquisition time.
    """

    correlations: list[np.ndarray] = []
    traces: list[np.ndarray | None] = []
    labels: list[str] = []

    def _add_block(kind: str, corr_block, time_block, trace_block) -> None:
        """Append one or more correlations of a given kind to the lists.

        ``corr_block`` and ``time_block`` are typically 1D or 2D NumPy
        arrays. For 1D correlations we create a single entry. For 2D
        correlations we create one entry per column.
        """

        tau = np.asarray(time_block, dtype=float).ravel()
        corr_arr = np.asarray(corr_block, dtype=float)

        # Single-column or 1D correlation
        if corr_arr.ndim == 1 or (corr_arr.ndim == 2 and corr_arr.shape[1] == 1):
            gg = corr_arr.reshape(-1)
            pair = np.empty((gg.size, 2), dtype=float)
            pair[:, 0] = tau
            pair[:, 1] = gg

            correlations.append(pair)
            labels.append(kind)

            if trace_block is not None:
                avg = float(np.asarray(trace_block, dtype=float).mean())
                # Minimal two-point trace: 0 s and 1 s with the same countrate.
                tr = np.zeros((2, 2), dtype=float)
                tr[:, 0] = [0.0, 1.0]
                tr[:, 1] = avg
                traces.append(tr)
            else:
                traces.append(None)
            return

        # Multi-column correlation (e.g. multiple channels)
        for j in range(corr_arr.shape[1]):
            gg = corr_arr[:, j]
            pair = np.empty((gg.size, 2), dtype=float)
            pair[:, 0] = tau
            pair[:, 1] = gg
            correlations.append(pair)
            labels.append(kind)

            if trace_block is not None:
                try:
                    avg = float(np.asarray(trace_block[j], dtype=float).mean())
                except Exception:
                    avg = float(np.asarray(trace_block, dtype=float).mean())
                tr = np.zeros((2, 2), dtype=float)
                tr[:, 0] = [0.0, 1.0]
                tr[:, 1] = avg
                traces.append(tr)
            else:
                traces.append(None)

    # Autocorrelations (AC)
    if "ac" in g_struct and "act" in g_struct:
        ac = g_struct["ac"]
        act = g_struct["act"]
        tr = g_struct.get("trace", None)

        # Work around the "single AC" layout where ac is a single long
        # vector instead of a list/array of vectors.
        if not isinstance(ac, (list, tuple, np.ndarray)) or (hasattr(ac, "ndim") and np.ndim(ac) == 1 and len(ac) > 4):
            ac_list = [np.asarray(ac)]
            act_list = [np.asarray(act)]
            trace_list = [tr] if tr is not None else [None]
        else:
            ac_list = list(ac)
            act_list = list(act) if isinstance(act, (list, tuple, np.ndarray)) else [act] * len(ac_list)
            if tr is None:
                trace_list = [None] * len(ac_list)
            else:
                # g['trace'] may be a list/array of averages per AC curve
                trace_list = list(tr) if isinstance(tr, (list, tuple, np.ndarray)) else [tr] * len(ac_list)

        for idx, (c_block, t_block, tr_block) in enumerate(zip(ac_list, act_list, trace_list)):
            _add_block(f"AC{idx + 1}", c_block, t_block, tr_block)

    # Dual-color correlations (DC)
    if "dc" in g_struct and "dct" in g_struct:
        dc = g_struct["dc"]
        dct = g_struct["dct"]
        dc_list = list(dc) if isinstance(dc, (list, tuple, np.ndarray)) else [dc]
        dct_list = list(dct) if isinstance(dct, (list, tuple, np.ndarray)) else [dct] * len(dc_list)

        for idx, (c_block, t_block) in enumerate(zip(dc_list, dct_list)):
            _add_block(f"CC dual color {idx + 1}", c_block, t_block, None)

    # Two-focus correlations
    if "twof" in g_struct and "twoft" in g_struct:
        twof = g_struct["twof"]
        twoft = g_struct["twoft"]
        tf_list = list(twof) if isinstance(twof, (list, tuple, np.ndarray)) else [twof]
        tft_list = list(twoft) if isinstance(twoft, (list, tuple, np.ndarray)) else [twoft] * len(tf_list)

        for idx, (c_block, t_block) in enumerate(zip(tf_list, tft_list)):
            _add_block(f"CC two foci {idx + 1}", c_block, t_block, None)

    # Dual-color two-focus correlations
    if "dc2f" in g_struct and "dc2ft" in g_struct:
        dc2f = g_struct["dc2f"]
        dc2ft = g_struct["dc2ft"]
        dcf_list = list(dc2f) if isinstance(dc2f, (list, tuple, np.ndarray)) else [dc2f]
        dcft_list = list(dc2ft) if isinstance(dc2ft, (list, tuple, np.ndarray)) else [dc2ft] * len(dcf_list)

        for idx, (c_block, t_block) in enumerate(zip(dcf_list, dcft_list)):
            _add_block(f"CC dual color two foci {idx + 1}", c_block, t_block, None)

    return correlations, traces, labels


def read_ries_mat(
        filename: str,
        verbose: bool = False
) -> typing.List[FCSDataset]:
    """Read Jonas Ries-style SFCS correlation .mat files.

    This reader re-implements the logic of the Ries SFCS tools in a
    ChiSurf-native way:

    - loads the MATLAB ``g`` structure from ``filename``;
    - builds correlation arrays and minimal intensity traces;
    - computes photon-noise weights using :func:`chisurf.fluorescence.fcs.noise`;
    - returns a list of :class:`FCSDataset` dictionaries.
    """

    path = pathlib.Path(filename)
    if verbose:
        print("Reading Ries .mat FCS file:", path)

    data = _load_nested_mat(path)
    if "g" not in data:
        raise ValueError(f"Ries .mat file '{filename}' does not contain a 'g' structure")

    g_struct = data["g"]
    corr_list, trace_list, type_list = _build_correlation_entries(g_struct, str(path))

    datasets: list[FCSDataset] = []

    for idx, corr in enumerate(corr_list):
        tau = np.asarray(corr[:, 0], dtype=float)
        amp = np.asarray(corr[:, 1], dtype=float)

        trace = trace_list[idx]
        # Very simple interpretation of the synthetic two-point traces:
        # acquisition time is the last time bin (1 s), countrate is the
        # average intensity in kHz.
        if trace is not None and trace.size >= 2:
            t_arr = np.asarray(trace, dtype=float)
            acq_time_s = float(t_arr[-1, 0])
            mean_cr_khz = float(np.mean(t_arr[:, 1]))
        else:
            acq_time_s = 1.0
            mean_cr_khz = 1.0

        # Compute Suren-style photon-noise standard deviations and convert
        # them into weights = 1/sigma.
        try:
            sd = chisurf.fluorescence.fcs.noise(
                times=tau,
                correlation=amp,
                measurement_duration=acq_time_s,
                mean_count_rate=mean_cr_khz,
            )
            w = np.ones_like(amp, dtype=float)
            tiny = 1e-12
            for i, v in enumerate(sd):
                if not np.isfinite(v) or abs(v) < tiny:
                    w[i] = 1.0
                else:
                    w[i] = 1.0 / float(v)
        except Exception:
            w = np.ones_like(amp, dtype=float)

        label = type_list[idx] if idx < len(type_list) else "AC"

        ds: FCSDataset = {  # type: ignore[assignment]
            "filename": str(path),
            "measurement_id": f"{path.stem}_{idx}",
            "acquisition_time": float(acq_time_s),
            "mean_count_rate": float(mean_cr_khz),
            "correlation_times": tau.tolist(),
            "correlation_amplitudes": amp.tolist(),
            "correlation_amplitude_weights": w.tolist(),
            "intensity_trace_times": trace[:, 0].tolist() if trace is not None else np.asarray([], dtype=float).tolist(),
            "intensity_trace": trace[:, 1].tolist() if trace is not None else np.asarray([], dtype=float).tolist(),
            "intensity_trace_name": label,
            "meta_data": {"ries_type": label},
        }
        datasets.append(ds)

    return datasets
