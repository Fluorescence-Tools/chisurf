from __future__ import annotations

import csv
import pathlib
from typing import List, Tuple, Union

import numpy as np

import chisurf as cs
from chisurf import typing
from chisurf.core.fio.fluorescence.fcs.definitions import FCSDataset


ArrayLike = np.ndarray
TraceSpec = Union[ArrayLike, List[ArrayLike], None]


def _detect_mode_tokens(lines: typing.List[str]) -> typing.List[str]:
    """Return the tokens after the first ``Mode=`` line.

    The correlator.com SIN formats encode either human-readable strings
    ("Single Auto", "Quad", ...) or integer channel indices on this
    line. We only need the tokens to decide between the two layouts.
    """

    for raw in lines:
        line = raw.strip()
        if line.lower().startswith("mode") and "=" in line:
            return line.split("=", 1)[1].strip().split()
    raise ValueError("No 'Mode=' line found in SIN file")


def _parse_integer_mode(lines: typing.List[str], mode_values: typing.List[int]) -> Tuple[List[ArrayLike], List[TraceSpec], List[str]]:
    """Parse correlator.com SIN files in *integer mode* layout.

    The integer mode encodes channel indices directly in ``Mode=`` and
    stores all correlations in a single block labeled
    ``[CorrelationFunction]``. Intensities are collected under
    ``[IntensityHistory]``.
    """

    corr_rows: typing.List[typing.List[str]] = []
    intensity_rows: typing.List[typing.List[str]] = []
    section = ""

    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if lower.startswith("["):
            section = lower
            continue
        if not line or "=" in line:
            continue
        if "[correlationfunction]" in section:
            corr_rows.append(line.split())
        elif "[intensityhistory]" in section:
            intensity_rows.append(line.split())

    if not corr_rows:
        raise ValueError("No [CorrelationFunction] section found in SIN integer-mode file")

    corr_arr = np.asarray(corr_rows, dtype=float)
    intensity_arr = np.asarray(intensity_rows, dtype=float) if intensity_rows else np.zeros((0, 0))

    # Convert to ms and kHz as in the original correlator.com reader.
    timefactor = 1000.0  # seconds -> milliseconds
    timedivfac = 1000.0  # Hz -> kHz

    corr_arr[:, 0] *= timefactor
    if intensity_arr.size:
        intensity_arr[:, 0] *= timefactor
        if intensity_arr.shape[1] > 1:
            intensity_arr[:, 1:] /= timedivfac

    # Correlator.com normalizes correlations to 1; convert to 0 baseline.
    if corr_arr.shape[1] > 1:
        corr_arr[:, 1:] -= 1.0

    correlations: List[ArrayLike] = []
    traces: List[TraceSpec] = []
    labels: List[str] = []

    if len(mode_values) % 2 != 0:
        raise ValueError("SIN integer-mode 'Mode' line must contain an even number of entries")

    n_pairs = len(mode_values) // 2
    for ii in range(n_pairs):
        mode_a = mode_values[2 * ii]
        mode_b = mode_values[2 * ii + 1]

        # Build correlation curve: lag time in ms, correlation value.
        corr = np.zeros((corr_arr.shape[0], 2), dtype=float)
        corr[:, 0] = corr_arr[:, 0]
        corr[:, 1] = corr_arr[:, ii + 1]
        correlations.append(corr)

        if intensity_arr.size == 0 or intensity_arr.shape[1] < 2:
            traces.append(None)
            labels.append(f"AC{mode_a}" if mode_a == mode_b else f"CC{mode_a}{mode_b}")
            continue

        if mode_a == mode_b:
            # Autocorrelation curve AC<i> with a single intensity trace.
            labels.append(f"AC{mode_a}")
            tr = np.zeros((intensity_arr.shape[0], 2), dtype=float)
            tr[:, 0] = intensity_arr[:, 0]
            # Column 0 is time, columns 1.. are channels.
            tr[:, 1] = intensity_arr[:, mode_a + 1]
            traces.append(tr)
        else:
            # Cross-correlation curve CC<i><j> with two intensity traces.
            labels.append(f"CC{mode_a}{mode_b}")
            modmin = min(mode_a, mode_b)
            modmax = max(mode_a, mode_b)
            tr_a = np.zeros((intensity_arr.shape[0], 2), dtype=float)
            tr_b = np.zeros((intensity_arr.shape[0], 2), dtype=float)
            tr_a[:, 0] = intensity_arr[:, 0]
            tr_b[:, 0] = intensity_arr[:, 0]
            tr_a[:, 1] = intensity_arr[:, modmin + 1]
            tr_b[:, 1] = intensity_arr[:, modmax + 1]
            traces.append([tr_a, tr_b])

    return correlations, traces, labels


def _parse_old_mode(lines: typing.List[str]) -> Tuple[List[ArrayLike], List[TraceSpec], List[str]]:
    """Parse the original text SIN layout used by correlator.com.

    Correlation data is provided in a ``[CorrelationFunction]`` block and
    intensity traces in ``[IntensityHistory]``. Depending on the
    acquisition mode, there can be one or two traces.
    """

    start_c = end_c = start_t = end_t = None
    mode_str = ""

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("Mode") and "=" in line:
            mode_str = line.split("=", 1)[1].strip()
        if line.startswith("[CorrelationFunction]"):
            start_c = idx + 1
        if line.startswith("[RawCorrelationFunction]"):
            end_c = idx - 2
        if line.startswith("[IntensityHistory]"):
            # One line with trace length follows before the data.
            start_t = idx + 2
        if line.startswith("[Histogram]"):
            end_t = idx - 2

    if start_c is None or end_c is None:
        raise ValueError("SIN file missing [CorrelationFunction] or [RawCorrelationFunction] section")
    if start_t is None or end_t is None:
        raise ValueError("SIN file missing [IntensityHistory] or [Histogram] section")

    corr_lines = lines[start_c:end_c]
    trace_lines = lines[start_t:end_t]

    timefactor = 1000.0  # seconds -> ms
    timedivfac = 1000.0  # Hz -> kHz

    readcorr = csv.reader(corr_lines, delimiter="\t")
    readtrace = csv.reader(trace_lines, delimiter="\t")

    correlations: List[ArrayLike] = []
    traces: List[TraceSpec] = []
    labels: List[str] = []

    mode_str = mode_str.strip()

    def _read_corr_columns(cols: typing.List[int]) -> typing.List[ArrayLike]:
        """Read full correlation columns from the CSV data.

        Parameters
        ----------
        cols : list of int
            Column indices to read.

        Returns
        -------
        list of ArrayLike
            List of correlation curve arrays.
        """
        buffers = [[] for _ in cols]
        for row in csv.reader(corr_lines, delimiter="\t"):
            if not row:
                continue
            try:
                tau_s = float(row[0])
            except (ValueError, IndexError):
                continue
            tau_ms = tau_s * timefactor
            for buf, ci in zip(buffers, cols):
                try:
                    val = float(row[ci]) - 1.0
                except (ValueError, IndexError):
                    val = 0.0
                buf.append((tau_ms, val))
        return [np.asarray(b, dtype=float) for b in buffers]

    def _read_traces(two_channels: bool) -> typing.List[ArrayLike]:
        """Read 1 or 2 intensity traces from the CSV data.

        Parameters
        ----------
        two_channels : bool
            If True, read two channels.

        Returns
        -------
        list of ArrayLike
            List of intensity trace arrays.
        """
        t1: typing.List[typing.Tuple[float, float]] = []
        t2: typing.List[typing.Tuple[float, float]] = []
        for row in csv.reader(trace_lines, delimiter="\t"):
            if not row:
                continue
            try:
                t_s = float(row[0])
            except (ValueError, IndexError):
                continue
            t_ms = t_s * timefactor
            try:
                i1 = float(row[1]) / timedivfac
            except (ValueError, IndexError):
                i1 = 0.0
            t1.append((t_ms, i1))
            if two_channels:
                try:
                    i2 = float(row[2]) / timedivfac
                except (ValueError, IndexError):
                    i2 = 0.0
                t2.append((t_ms, i2))
        out: typing.List[ArrayLike] = []
        out.append(np.asarray(t1, dtype=float))
        if two_channels:
            out.append(np.asarray(t2, dtype=float))
        return out

    if mode_str == "Single Auto":
        labels.append("AC")
        corr_list = _read_corr_columns([1])
        correlations.extend(corr_list)
        trace_arr = _read_traces(two_channels=False)[0]
        traces.append(trace_arr)
    elif mode_str == "Single Cross":
        labels.append("CC")
        corr_list = _read_corr_columns([1])
        correlations.extend(corr_list)
        t1, t2 = _read_traces(two_channels=True)
        traces.append([t1, t2])
    elif mode_str == "Dual Auto":
        labels.extend(["AC1", "AC2"])
        corr_list = _read_corr_columns([1, 2])
        correlations.extend(corr_list)
        t1, t2 = _read_traces(two_channels=True)
        traces.extend([t1, t2])
    elif mode_str == "Dual Cross":
        labels.extend(["CC12", "CC21"])
        corr_list = _read_corr_columns([1, 2])
        correlations.extend(corr_list)
        t1, t2 = _read_traces(two_channels=True)
        # For cross-correlation each curve conceptually "sees" both traces.
        traces.extend([[t1, t2], [t1, t2]])
    elif mode_str == "Quad":
        labels.extend(["AC1", "AC2", "CC12", "CC21"])
        corr_list = _read_corr_columns([1, 2, 3, 4])
        correlations.extend(corr_list)
        t1, t2 = _read_traces(two_channels=True)
        traces.extend([t1, t2, [t1, t2], [t1, t2]])
    else:
        raise NotImplementedError(f"SIN mode '{mode_str}' is not supported")

    return correlations, traces, labels


def _estimate_trace_stats(trace: TraceSpec) -> typing.Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (acquisition_time_s, mean_count_rate_khz, t_ms, i_khz)."""

    if trace is None:
        return 1.0, 1.0, np.asarray([], dtype=float), np.asarray([], dtype=float)

    if isinstance(trace, list):
        # Multiple traces (e.g. cross-correlation): combine intensities.
        arrays = [np.asarray(t, dtype=float) for t in trace if np.asarray(t).size]
        if not arrays:
            return 1.0, 1.0, np.asarray([], dtype=float), np.asarray([], dtype=float)
        base = arrays[0]
        t_ms = base[:, 0]
        i_khz = np.zeros_like(base[:, 1])
        for arr in arrays:
            if arr.shape[0] != base.shape[0]:
                continue
            i_khz += arr[:, 1]
    else:
        arr = np.asarray(trace, dtype=float)
        if arr.size == 0 or arr.shape[1] < 2:
            return 1.0, 1.0, np.asarray([], dtype=float), np.asarray([], dtype=float)
        t_ms = arr[:, 0]
        i_khz = arr[:, 1]

    if t_ms.size == 0:
        return 1.0, 1.0, t_ms, i_khz

    acq_s = float(t_ms[-1] - t_ms[0]) / 1000.0
    if acq_s <= 0:
        acq_s = float(t_ms[-1]) / 1000.0
    if acq_s <= 0:
        acq_s = 1.0

    mean_cr_khz = float(np.mean(i_khz)) if i_khz.size else 1.0
    return acq_s, mean_cr_khz, t_ms, i_khz


def read_sin(
        filename: str,
        verbose: bool = False
) -> typing.List[FCSDataset]:
    """Read correlator.com ``.SIN`` FCS files into ChiSurf.

    This is a native implementation based on the documented correlator.com
    layouts and the behavior of the PyCorrFit ``read_SIN_correlator_com``
    helper, but it does **not** depend on PyCorrFit at runtime.

    The function returns a list of :class:`FCSDataset` entries, one per
    correlation curve contained in the file. Correlation lag times are
    stored in milliseconds, intensities in kHz, and the acquisition time
    is recorded in seconds so that :func:`cs.core.fluorescence.fcs.noise`
    or :func:`cs.core.fluorescence.fcs.compute_weights` can be applied
    consistently.
    """

    path = pathlib.Path(filename)
    if verbose:
        print("Reading correlator.com SIN file:", path)

    text = path.read_text(encoding="latin1").splitlines()

    tokens = _detect_mode_tokens(text)
    # All-digit tokens indicate the integer mode layout.
    if all(tok.isdigit() for tok in tokens):
        mode_vals = [int(t) for t in tokens]
        corr_list, trace_list, type_list = _parse_integer_mode(text, mode_vals)
    else:
        corr_list, trace_list, type_list = _parse_old_mode(text)

    datasets: typing.List[FCSDataset] = []

    for idx, corr in enumerate(corr_list):
        arr = np.asarray(corr, dtype=float)
        if arr.size == 0 or arr.shape[1] < 2:
            continue
        tau_ms = arr[:, 0]
        g = arr[:, 1]

        trace_spec = trace_list[idx] if idx < len(trace_list) else None
        acq_s, mean_cr_khz, t_ms, i_khz = _estimate_trace_stats(trace_spec)

        label = type_list[idx] if idx < len(type_list) else f"curve{idx}"

        ds: FCSDataset = {  # type: ignore[assignment]
            "filename": str(path),
            "measurement_id": f"{path.stem}_{idx}",
            "acquisition_time": float(acq_s),
            "mean_count_rate": float(mean_cr_khz),
            "correlation_times": tau_ms.tolist(),
            "correlation_amplitudes": g.tolist(),
            # Let read_fcs attach or recompute weights if needed.
            "correlation_amplitude_weights": np.ones_like(g, dtype=float).tolist(),
            "intensity_trace_times": t_ms.tolist(),
            "intensity_trace": i_khz.tolist(),
            "intensity_trace_name": label,
            "meta_data": {"sin_mode": label},
        }
        datasets.append(ds)

    return datasets
