from __future__ import annotations

"""CSV / whitespace-delimited DEER trace loader (numpy-only).

Accepts two- or three-column text files ``t, V_real[, V_imag]`` (comma,
semicolon, tab or whitespace separated), skipping comment/header lines, plus
plain ASCII exports. Time is converted to microseconds from an inferred unit.
"""

import re

import numpy as np

#: Multipliers converting a detected time unit to microseconds.
_TIME_UNIT_US = {"ns": 1e-3, "us": 1.0, "µs": 1.0, "ms": 1e3, "s": 1e6}


def _detect_time_unit(header: str) -> float | None:
    """Return the µs multiplier inferred from a header line, or ``None``.

    ``None`` means no unit was stated and the caller should fall back to a
    magnitude heuristic on the actual time values.
    """
    h = header.lower()
    for unit, mult in _TIME_UNIT_US.items():
        if f"[{unit}]" in h or f"({unit})" in h or f" {unit}" in h:
            return mult
    return None


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a DEER trace from a delimited text file.

    Returns
    -------
    (t, v_real, v_imag, attrs)
        Time in microseconds and the real/imaginary signal columns
        (``v_imag`` is zeros when absent).
    """
    header = ""
    rows: list[list[float]] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s[0] in "#%" or re.match(r"^[A-Za-z]", s):
                header = s
                continue
            parts = re.split(r"[,;\t ]+", s)
            try:
                vals = [float(p) for p in parts if p != ""]
            except ValueError:
                header = s
                continue
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric data found in {path!r}")

    arr = np.array([r[: max(len(x) for x in rows)] + [np.nan] * (max(len(x) for x in rows) - len(r))
                    for r in rows], dtype=float)
    t_raw = arr[:, 0]
    mult = _detect_time_unit(header)
    if mult is None:
        # No stated unit: DEER traces span a few µs, so a span of many hundreds
        # is almost certainly nanoseconds. Fall back to µs for small spans.
        span = float(np.nanmax(np.abs(t_raw)) - np.nanmin(t_raw))
        mult = 1e-3 if span > 50.0 else 1.0
    t = t_raw * mult
    v_real = arr[:, 1] if arr.shape[1] > 1 else np.zeros_like(t)
    v_imag = arr[:, 2] if arr.shape[1] > 2 else np.zeros_like(t)
    v_imag = np.nan_to_num(v_imag)
    return t, v_real, v_imag, {"time_unit_us": mult}
