"""Core algorithms for TTTR microtime LUT computation."""

from __future__ import annotations

import glob
import os

import click
import numpy as np

try:
    import tttrlib
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("ERROR: tttrlib is required. Install from conda/pip.") from exc


def expand_globs(patterns: list[str]) -> list[str]:
    """Expand file glob patterns and return unique existing paths.

    Parameters
    ----------
    patterns : list of str
        Glob patterns to expand.

    Returns
    -------
    list of str
        Unique matching paths in input order.
    """
    files: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            click.echo(f"WARNING: pattern matched no files: {pattern}", err=True)
        files.extend(matches)

    unique: list[str] = []
    seen: set[str] = set()
    for filename in files:
        if filename not in seen:
            seen.add(filename)
            unique.append(filename)
    return unique


def load_microtimes(file_list: list[str]) -> np.ndarray:
    """Load and concatenate microtime arrays from TTTR files.

    Parameters
    ----------
    file_list : list of str
        TTTR file paths.

    Returns
    -------
    numpy.ndarray
        Concatenated microtime values.
    """
    parts: list[np.ndarray] = []
    for filename in file_list:
        click.echo(f"Loading {filename}")
        tttr = tttrlib.TTTR(filename)
        microtimes = tttr.micro_times
        if microtimes is None or len(microtimes) == 0:
            click.echo(f"  WARNING: no microtimes in {filename}", err=True)
            continue
        click.echo(f"  {len(microtimes):,} events")
        parts.append(microtimes)

    if not parts:
        raise RuntimeError("No microtimes found in any input file.")

    all_micro = np.concatenate(parts)
    click.echo(f"Total microtimes: {len(all_micro):,}")
    return all_micro


def infer_n_bins(micro: np.ndarray, n_bins_opt: int | None) -> int:
    """Infer a TAC histogram bin count from microtime data.

    Parameters
    ----------
    micro : numpy.ndarray
        Raw microtime values.
    n_bins_opt : int or None
        User-specified bin count. If positive, it is returned directly.

    Returns
    -------
    int
        Histogram bin count.
    """
    if n_bins_opt and n_bins_opt > 0:
        return int(n_bins_opt)

    n_bins = int(np.max(micro)) + 1
    for nice_bins in (4096, 8192, 16384, 32768, 65536):
        if abs(n_bins - nice_bins) <= max(8, int(0.002 * nice_bins)):
            n_bins = nice_bins
            break

    click.echo(f"[auto] inferred n_bins = {n_bins}")
    return n_bins


def histogram_micro(micro: np.ndarray, n_bins: int) -> np.ndarray:
    """Build a TAC histogram from microtime values.

    Parameters
    ----------
    micro : numpy.ndarray
        Raw microtime values.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    numpy.ndarray
        Histogram counts.
    """
    counts, _ = np.histogram(micro, bins=n_bins, range=(0, n_bins))
    return counts


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Return a padded rolling mean for a 1D array.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    win : int
        Rolling window size.

    Returns
    -------
    numpy.ndarray
        Rolling mean with edge padding.
    """
    if win <= 1:
        return x.astype(float)

    cumulative = np.cumsum(np.insert(x, 0, 0))
    out = (cumulative[win:] - cumulative[:-win]) / float(win)
    pad_left = win // 2
    pad_right = len(x) - len(out) - pad_left
    return np.pad(out, (pad_left, pad_right), mode="edge")


def find_longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Find the longest contiguous run of True values.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean mask.

    Returns
    -------
    tuple of int or None
        Half-open ``(start, stop)`` interval, or ``None`` when no True run exists.
    """
    best_len = 0
    best_start = -1
    cur_len = 0
    cur_start = -1

    for index, value in enumerate(mask):
        if value:
            if cur_len == 0:
                cur_start = index
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_len == 0:
        return None
    return best_start, best_start + best_len


def autodetect_linear_region(
    counts: np.ndarray,
    noffset_guess: int = 0,
    tail_exclude_frac: float = 0.0,
    win: int = 31,
    rel_dev_thresh: float = 0.10,
    slope_thresh: float = 0.02,
    min_width: int = 32,
) -> tuple[int, int]:
    """Detect a stable TAC plateau for LUT linearization.

    Parameters
    ----------
    counts : numpy.ndarray
        TAC histogram counts.
    noffset_guess : int, default=0
        Initial offset guess used to skip early bins.
    tail_exclude_frac : float, default=0.0
        Fraction of the tail to exclude.
    win : int, default=31
        Rolling-window size for stability estimates.
    rel_dev_thresh : float, default=0.10
        Maximum relative deviation from the rolling mean.
    slope_thresh : float, default=0.02
        Maximum relative rolling-mean slope.
    min_width : int, default=32
        Minimum plateau width.

    Returns
    -------
    tuple of int
        Half-open ``(linear_start, linear_stop)`` plateau interval.

    Raises
    ------
    ValueError
        If no sufficiently stable plateau is found.
    """
    n_bins = len(counts)
    lo = int(np.clip(noffset_guess, 0, n_bins - 2))
    hi = int(np.clip(n_bins - int(n_bins * tail_exclude_frac), lo + 1, n_bins))

    region_counts = counts[lo:hi].astype(float)
    mean = rolling_mean(region_counts, win=win)
    mean = np.where(mean <= 0, 1.0, mean)

    rel_dev = np.abs(region_counts / mean - 1.0)
    rel_slope = np.abs(np.gradient(mean) / np.maximum(mean, 1.0))
    stable = (rel_dev <= rel_dev_thresh) & (rel_slope <= slope_thresh)
    run = find_longest_true_run(stable)
    if run is None:
        raise ValueError("no plateau")

    start_rel, stop_rel = run
    if (stop_rel - start_rel) < min_width:
        raise ValueError(f"plateau too short ({stop_rel - start_rel} < {min_width})")

    return lo + start_rel, lo + stop_rel


def build_linearization_table(
    counts: np.ndarray,
    linear_start: int,
    linear_stop: int,
    ntac_required: int,
    noffset: int,
) -> dict[str, int | float | np.ndarray]:
    """Build the Felekyan-style TAC linearization table.

    Parameters
    ----------
    counts : numpy.ndarray
        Effective TAC histogram counts.
    linear_start : int
        First bin of the selected linear region.
    linear_stop : int
        First bin after the selected linear region.
    ntac_required : int
        Desired number of NTAC bins.
    noffset : int
        Offset subtracted from corrected NTAC indices.

    Returns
    -------
    dict
        LUT table containing cumulative NTAC fractions and metadata.
    """
    region = counts[linear_start:linear_stop]
    if region.sum() == 0:
        raise ValueError("Chosen linear region has zero counts.")

    mean = float(region.mean())
    widths = counts.astype(np.float64) / (mean if mean != 0 else 1.0)
    cumulative = np.cumsum(widths)
    total = cumulative[-1]
    if total <= 0:
        raise ValueError("Cumulative width is zero.")

    scale = float(ntac_required) / float(total)
    ntac_fract = scale * cumulative

    return {
        "NTAC_fract": ntac_fract,
        "w": widths,
        "f": scale,
        "n_bins": int(len(counts)),
        "linear_start": int(linear_start),
        "linear_stop": int(linear_stop),
        "ntac_required": int(ntac_required),
        "noffset": int(noffset),
        "n_mean": float(mean),
        "total_counts": int(counts.sum()),
    }


def stochastic_rebin_ntac(
    raw_micro: np.ndarray,
    ntac_fract: np.ndarray,
    noffset: int,
    seed: int | None = None,
    max_photons: int | None = None,
    rounding: str = "ceil",
    eps: float = 0.0,
) -> np.ndarray:
    """Apply a LUT to raw microtimes and return corrected NTAC indices.

    Parameters
    ----------
    raw_micro : numpy.ndarray
        Raw microtime values.
    ntac_fract : numpy.ndarray
        Cumulative NTAC fractions.
    noffset : int
        Offset subtracted from corrected indices.
    seed : int or None, optional
        Random seed for stochastic correction.
    max_photons : int or None, optional
        Maximum number of photons to process for previews.
    rounding : {"ceil", "floor", "stochastic"}, default="ceil"
        Rounding mode.
    eps : float, default=0.0
        Small epsilon used to avoid mapping exactly to the last boundary.

    Returns
    -------
    numpy.ndarray
        Corrected integer NTAC indices.
    """
    rng = np.random.default_rng(seed)
    if eps and eps > 0:
        ntac_fract = np.array(ntac_fract, copy=True)
        ntac_fract[-1] = np.nextafter(ntac_fract[-1] - float(eps), -np.inf)

    n_bins = ntac_fract.shape[0]
    raw = raw_micro.astype(np.int64)
    if max_photons is not None and max_photons > 0:
        raw = raw[:max_photons]
    raw = np.clip(raw, 0, n_bins - 1)

    left = np.zeros_like(raw, dtype=np.float64)
    right = ntac_fract[raw].astype(np.float64)
    mask = raw > 0
    left[mask] = ntac_fract[raw[mask] - 1]

    span = right - left
    span = np.where(span > 0, span, 0.0)

    u = rng.random(raw.shape[0])
    frac_pos = left + u * span

    if rounding == "floor":
        ntac_int = np.floor(frac_pos).astype(np.int64)
    elif rounding == "stochastic":
        ntac_int = np.floor(frac_pos + rng.random(frac_pos.shape)).astype(np.int64)
    else:
        ntac_int = np.ceil(frac_pos).astype(np.int64)

    return ntac_int - noffset


def save_lut(path: str, table: dict[str, object]) -> None:
    """Save a LUT table to disk.

    Parameters
    ----------
    path : str
        Output path. Supported extensions are ``.txt``, ``.csv``, ``.npy``, and ``.npz``.
    table : dict
        LUT table produced by :func:`build_linearization_table`.
    """
    abs_path = os.path.abspath(path)
    ext = os.path.splitext(abs_path)[1].lower()
    ntac_fract = table["NTAC_fract"]

    if ext == ".txt":
        np.savetxt(abs_path, np.asarray(ntac_fract).reshape(-1, 1), fmt="%.9f", header="NTAC_fract")
    elif ext == ".csv":
        np.savetxt(abs_path, np.asarray(ntac_fract).reshape(-1, 1), delimiter=",", fmt="%.9f", header="NTAC_fract")
    elif ext == ".npy":
        np.save(abs_path, ntac_fract)
    elif ext == ".npz":
        np.savez_compressed(abs_path, **table)
    else:
        raise click.UsageError("Unknown output extension. Use .txt / .csv / .npy / .npz.")

    click.echo(f"Saved {ext.upper()}: {abs_path}")
