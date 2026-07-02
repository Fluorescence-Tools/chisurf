"""Per-pixel imaging maps: Number & Brightness (N&B) and phasor.

Qt-free helpers shared by the imaging N&B and phasor plugins. tttrlib is
imported lazily so this module stays importable (and partially testable) where
the compiled extension is absent.

- :func:`nb_maps` computes the per-pixel Number & Brightness moments from an
  intensity frame-stack (the one analysis PAM/tttrlib did not already provide).
- :func:`phasor_maps` is a thin wrapper over the built-in
  ``tttrlib.CLSMImage.get_phasor`` (raw + optional IRF reference).
- :func:`maps_to_dataframe` / :func:`write_imaging_hdf5` serialise per-pixel maps
  in the same layout the pixel-wise MLE writes (pandas ``key='results'`` table),
  which is also directly readable by ndxplorer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_stack(intensity: Any) -> np.ndarray:
    """Return an intensity image as a ``(n_frames, n_lines, n_pixel)`` stack."""
    arr = np.asarray(intensity, dtype=float)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported intensity shape {arr.shape!r}; expected 2D or 3D")


def nb_maps(intensity_stack: Any) -> dict[str, np.ndarray]:
    """Return per-pixel Number & Brightness maps from an intensity frame-stack.

    For each pixel, over the frame axis (the temporal samples):

    - ``mean``     = <k>
    - ``variance`` = <k²> − <k>²  (population variance across frames)
    - ``B``        = variance / mean            (apparent brightness)
    - ``N``        = mean² / variance           (apparent number)
    - ``epsilon``  = B − 1                       (true molecular brightness,
      photon-counting detector assumption)

    Parameters
    ----------
    intensity_stack : array_like
        ``(n_frames, n_lines, n_pixel)`` (or a single 2D frame) of photon counts.

    Returns
    -------
    dict of numpy.ndarray
        2-D maps ``mean``, ``variance``, ``B``, ``N``, ``epsilon``. Undefined
        pixels (zero mean/variance, or < 2 frames) are set to 0.
    """
    stack = _ensure_stack(intensity_stack)
    mean = stack.mean(axis=0)
    if stack.shape[0] < 2:
        variance = np.zeros_like(mean)
    else:
        variance = stack.var(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(mean > 0.0, variance / mean, 0.0)
        N = np.where(variance > 0.0, mean * mean / variance, 0.0)
    epsilon = np.where(mean > 0.0, B - 1.0, 0.0)
    return {
        "mean": np.nan_to_num(mean),
        "variance": np.nan_to_num(variance),
        "B": np.nan_to_num(B),
        "N": np.nan_to_num(N),
        "epsilon": np.nan_to_num(epsilon),
    }


def intensity_maps(intensity_stack: Any) -> dict[str, np.ndarray]:
    """Return the summed per-pixel intensity map ``{"intensity": ...}``."""
    stack = _ensure_stack(intensity_stack)
    return {"intensity": stack.sum(axis=0)}


def micro_time_channels_per_period(tttr: Any) -> int:
    """Return the number of micro-time channels in **one laser period**.

    The header's ``number_of_micro_time_channels`` is the TAC/ADC bit-depth range
    (e.g. 32768), usually larger than the laser period. The physically correct
    length is ``laser_period / micro_time_resolution`` = ``macro_time_resolution
    / micro_time_resolution`` (macro-time ticks are the laser sync). Falls back to
    the header value when the resolutions are unavailable. Capped at the ADC range.
    """
    header = tttr.header
    adc = int(getattr(header, "number_of_micro_time_channels", 0) or 0)
    micro_res = float(getattr(header, "micro_time_resolution", 0.0) or 0.0)
    macro_res = float(getattr(header, "macro_time_resolution", 0.0) or 0.0)
    if micro_res > 0.0 and macro_res > 0.0:
        n = int(round(macro_res / micro_res))
        if adc > 0:
            n = min(n, adc)
        return max(1, n)
    return max(1, adc or 1)


def micro_time_histogram(tttr: Any, channels=None, n_channels: int | None = None) -> np.ndarray:
    """Return the micro-time (TAC) histogram for *channels*, one laser period long.

    The length is derived from the rep rate and micro-time resolution
    (:func:`micro_time_channels_per_period`), not from the (larger) ADC range or
    the highest occupied bin — so the decay spans exactly one laser period
    regardless of which bins happen to be empty. Photons past one period
    (jitter/after-pulses) are dropped.

    *n_channels* forces the length (e.g. to align an IRF histogram to the data's
    period so they share one micro-time axis).
    """
    micro = np.asarray(tttr.micro_times)
    if channels:
        idx = np.asarray(tttr.get_selection_by_channel([int(c) for c in channels]))
        micro = micro[idx]
    n = int(n_channels) if n_channels else micro_time_channels_per_period(tttr)
    n = max(1, n)
    if micro.size == 0:
        return np.zeros(n, dtype=float)
    return np.bincount(micro, minlength=n)[:n].astype(float)


def build_clsm(tttr: Any, channels=(0,)) -> Any:
    """Build a filled ``tttrlib.CLSMImage`` (markers auto-detected from header)."""
    import tttrlib

    return tttrlib.CLSMImage(tttr, channels=list(channels), fill=True)


# --- shared TTTR / CLSM cache -------------------------------------------------
# Bounded caches so the (expensive) TTTR load and per-window CLSM fills are done
# ONCE and reused across the imaging tools (Intensity/N&B/phasor) and across
# step revisits, instead of each plugin rebuilding them.
import logging as _logging  # noqa: E402
import threading  # noqa: E402
from collections import OrderedDict  # noqa: E402

_logger = _logging.getLogger(__name__)
_TTTR_CACHE: OrderedDict[str, Any] = OrderedDict()
_CLSM_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_CACHE_LOCK = threading.RLock()  # cache is touched by the UI + a prefill thread
_TTTR_MAX = 3
_CLSM_MAX = 16


def get_tttr(filename: str) -> Any:
    """Return a cached ``tttrlib.TTTR`` for *filename* (loads once per session)."""
    key = str(filename)
    with _CACHE_LOCK:
        tttr = _TTTR_CACHE.get(key)
        if tttr is not None:
            _TTTR_CACHE.move_to_end(key)
            return tttr
    # Load outside the lock (slow) so a prefill load never blocks the UI thread.
    import tttrlib

    tttr = tttrlib.TTTR(key)
    with _CACHE_LOCK:
        _TTTR_CACHE[key] = tttr
        while len(_TTTR_CACHE) > _TTTR_MAX:
            _TTTR_CACHE.popitem(last=False)
    return tttr


def cached_clsm(filename: str, channels, micro_time_ranges=None) -> Any:
    """Return a cached windowed CLSM for ``(filename, channels, micro_time_ranges)``.

    Shared across the imaging tools (and warmed by the background prefill) so a
    given window's image is built once. Thread-safe: the fill runs outside the
    lock so a background prefill and the UI never block on each other for the
    dict (only the GIL serialises the actual C++ fill).
    """
    chs = tuple(int(x) for x in (channels or [0]))
    mtr = tuple(tuple(int(v) for v in r) for r in (micro_time_ranges or []))
    key = (str(filename), chs, mtr)
    with _CACHE_LOCK:
        clsm = _CLSM_CACHE.get(key)
        if clsm is not None:
            _CLSM_CACHE.move_to_end(key)
            return clsm
    clsm = build_clsm_windowed(get_tttr(filename), list(chs), list(mtr))
    with _CACHE_LOCK:
        _CLSM_CACHE[key] = clsm
        while len(_CLSM_CACHE) > _CLSM_MAX:
            _CLSM_CACHE.popitem(last=False)
    return clsm


def prefill_windows(filename: str, windows: dict) -> None:
    """Warm the CLSM cache for every window's channel-sets (background prefill).

    Builds the fills needed by the Intensity (``ch_p``/``ch_s``) and N&B/phasor
    (``chs``) steps, so that when the user advances the maps are already cached.
    """
    seen: set = set()
    for det in windows.values():
        mtr = det.get("micro_time_ranges") or []
        channel_sets = [det.get("chs", [0]) or [0]]
        ch_p, ch_s = detector_ps_channels(det)
        if ch_p:
            channel_sets.append(ch_p)
        if ch_s:
            channel_sets.append(ch_s)
        for chs in channel_sets:
            key = (tuple(chs), tuple(tuple(r) for r in mtr))
            if key in seen:
                continue
            seen.add(key)
            try:
                cached_clsm(filename, list(chs), list(mtr))
            except Exception:
                _logger.debug("prefill of %r failed", chs, exc_info=True)


def clear_imaging_cache() -> None:
    """Drop all cached TTTR/CLSM objects (free memory / force a fresh rebuild)."""
    with _CACHE_LOCK:
        _TTTR_CACHE.clear()
        _CLSM_CACHE.clear()


# --- worker process for GIL-holding reads -------------------------------------
# The tttrlib file read (``TTTR(path)``) holds the GIL for the bulk of the read,
# so binning on a background *thread* still freezes the UI. A worker *process*
# does the read + binning; only the small histogram arrays cross back. Spawn is
# safe here because this module is Qt-free (importing it in the child does not
# pull in the Qt stack), so the child stays lightweight.
_PROC_POOL: Any = None
_PROC_POOL_LOCK = threading.Lock()


def _get_proc_pool():
    """Return the lazily-created single-worker spawn process pool."""
    global _PROC_POOL
    with _PROC_POOL_LOCK:
        if _PROC_POOL is None:
            import concurrent.futures as _futures
            import multiprocessing as _mp

            _PROC_POOL = _futures.ProcessPoolExecutor(
                max_workers=1, mp_context=_mp.get_context("spawn")
            )
        return _PROC_POOL


def shutdown_proc_pool() -> None:
    """Shut down the worker process (best-effort; frees the child's TTTR cache)."""
    global _PROC_POOL
    with _PROC_POOL_LOCK:
        if _PROC_POOL is not None:
            try:
                _PROC_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                _logger.debug("proc pool shutdown failed", exc_info=True)
            _PROC_POOL = None


def _calibration_hist_worker(filename, chs, ch_p, ch_s, irf_files):
    """Read + bin the calibration decay/IRF histograms (runs in the worker process)."""
    tttr = get_tttr(filename)
    data = micro_time_histogram(tttr, chs)
    n = len(data)
    data_vv = micro_time_histogram(tttr, ch_p or chs, n_channels=n)
    data_vh = micro_time_histogram(tttr, ch_s, n_channels=n) if ch_s else np.zeros(n)
    raw = (
        raw_irf_components(list(irf_files), ch_p or chs, ch_s, n_channels=n)
        if irf_files else {"vv": None, "vh": None}
    )
    return {
        "data": data, "data_vv": data_vv, "data_vh": data_vh,
        "irf_vv_raw": raw["vv"], "irf_vh_raw": raw["vh"], "n": n,
    }


def calibration_histograms(filename, chs, ch_p, ch_s, irf_files) -> dict:
    """Compute the calibration decay/IRF histograms in a worker process.

    The tttrlib file read holds the GIL, so a background *thread* still freezes
    the UI; a worker process keeps the caller's UI fully responsive and only the
    small histogram arrays cross the process boundary. Falls back to an in-process
    read if the pool is unavailable (e.g. a platform without spawn).
    """
    args = (str(filename), list(chs or []), list(ch_p or []), list(ch_s or []), list(irf_files or []))
    try:
        return _get_proc_pool().submit(_calibration_hist_worker, *args).result()
    except Exception:
        _logger.debug("calibration histogram worker failed; falling back in-process", exc_info=True)
        shutdown_proc_pool()  # drop a broken pool so the next call recreates it
        return _calibration_hist_worker(*args)


# --- per-window compute -------------------------------------------------------
# Runs in a worker PROCESS (see ``compute_windows``): both the TTTR read and the
# CLSM fill hold the GIL, so a background *thread* still freezes the UI — only a
# separate process keeps it responsive. Spawn is cheap here because this module
# is Qt-free (the child does NOT re-import the Qt stack). A single worker keeps
# the read/fill cached across windows; only the resulting arrays cross back.
def _window_worker(payload):
    """Compute one window's maps (with optional per-detector IRF / BG calibration).

    Every result also carries a display-only per-frame intensity stack under
    ``"frames"`` (and, for phasor, per-frame ``"g_frames"`` / ``"s_frames"``) so
    the movie docks never rebuild a CLSM on the UI/pipeline thread. Run in a
    worker process (:func:`compute_windows`) because the CLSM fill holds the GIL.
    """
    filename, chs, ch_p, ch_s, mtr, kind, params, irf_files, bg = payload
    clsm = cached_clsm(filename, chs, mtr)
    if kind == "nb":
        stack = np.asarray(clsm.get_intensity(), dtype=float)
        nb = nb_maps(stack)
        return {
            "intensity": stack.sum(axis=0), "N": nb["N"], "B": nb["B"],
            "epsilon": nb["epsilon"], "frames": stack,
        }
    if kind == "mean_micro_time":
        tttr = get_tttr(filename)
        stack = np.asarray(clsm.get_intensity(), dtype=float)
        # Resolve the micro-time resolution (s → ns) so the map is in nanoseconds;
        # -1 keeps tttrlib's default (units of the micro-time channel).
        res_ns = float(params.get("microtime_resolution", -1.0) or -1.0)
        if res_ns < 0.0:
            micro_res = float(getattr(tttr.header, "micro_time_resolution", 0.0) or 0.0)
            res_ns = micro_res * 1e9 if micro_res > 0.0 else -1.0
        n_ph = int(params.get("n_ph_min", 2))
        mt = np.asarray(clsm.get_mean_micro_time(tttr, res_ns, n_ph, True), dtype=float)
        # stack_frames=True → a single (1, n_lines, n_pixel) frame.
        if mt.ndim == 3:
            mt = mt[0]
        # Unstacked (per-frame) stack for the movie dock — computed here in the
        # worker so the movie never rebuilds a CLSM on the UI/pipeline thread.
        mt_frames = np.nan_to_num(
            np.asarray(clsm.get_mean_micro_time(tttr, res_ns, n_ph, False), dtype=float)
        )
        return {
            "intensity": stack.sum(axis=0), "mean_micro_time": np.nan_to_num(mt),
            "frames": stack, "mt_frames": mt_frames,
        }
    if kind == "phasor":
        stack = np.asarray(clsm.get_intensity(), dtype=float)
        # Per-detector IRF list (from the calibration step) takes precedence.
        irf_file = (irf_files[0] if irf_files else None) or params.get("irf")
        irf = get_tttr(irf_file) if irf_file else None
        tttr = get_tttr(filename)
        freq = params.get("frequency", -1.0)
        n_ph = params.get("n_ph_min", 2)
        ph = phasor_maps(clsm, tttr, frequency=freq, tttr_irf=irf, n_ph_min=n_ph)
        pf = phasor_frames(clsm, tttr, frequency=freq, tttr_irf=irf, n_ph_min=n_ph)
        return {
            "intensity": stack.sum(axis=0), "g": ph["g"], "s": ph["s"],
            "n_photons": ph["n_photons"], "frames": stack,
            "g_frames": pf["g"], "s_frames": pf["s"],
        }
    if kind == "intensity":
        base_chs = ch_p or chs
        base = cached_clsm(filename, base_chs, mtr)
        n_par = np.asarray(base.get_intensity(), dtype=float).sum(axis=0)
        if ch_s and ch_s != base_chs:
            n_perp = np.asarray(cached_clsm(filename, ch_s, mtr).get_intensity(), dtype=float).sum(axis=0)
        else:
            n_perp = np.zeros_like(n_par)
        durations, n_pixel = total_line_durations(base)
        return {
            "n_par": n_par, "n_perp": n_perp, "durations": durations,
            "n_pixel": n_pixel, "bg": float(bg),
            "frames": np.asarray(clsm.get_intensity(), dtype=float),
        }
    raise ValueError(f"unknown window kind {kind!r}")


def compute_windows(filename, windows, kind, params=None, progress=None, use_process=True):
    """Compute every window's maps (cached, sequential).

    Returns ``{window_name: {key: 2-D array (or scalar)}}`` and reports per-window
    progress so a progress dialog advances immediately. Each window dict may carry
    per-detector calibration: ``irf`` (file path) and ``bg`` (background kHz).

    The per-window work (TTTR read + CLSM fill) both hold the GIL, so it runs in a
    **worker process** (``use_process``): only the resulting arrays cross back, and
    the caller's UI thread stays fully responsive. A single worker keeps the read
    cached across windows; it falls back in-process if the pool is unavailable.
    """
    params = params or {}
    items = list(windows.items())
    n = len(items)
    results: dict = {}
    pool = _get_proc_pool() if use_process else None
    for i, (name, det) in enumerate(items):
        if callable(progress):
            progress(i / max(n, 1), f"Window {name} ({i + 1}/{n})")
        payload = (
            filename, list(det.get("chs", [0]) or [0]), list(det.get("ch_p") or []),
            list(det.get("ch_s") or []), list(det.get("micro_time_ranges") or []), kind, params,
            list(det.get("irf") or []), float(det.get("bg") or 0.0),
        )
        if pool is not None:
            try:
                results[name] = pool.submit(_window_worker, payload).result()
                continue
            except Exception:
                _logger.debug("window worker (process) failed; in-process fallback", exc_info=True)
                shutdown_proc_pool()
                pool = None
        results[name] = _window_worker(payload)
    if callable(progress):
        progress(1.0, "Done")
    return results


def build_clsm_windowed(tttr: Any, channels, micro_time_ranges=None) -> Any:
    """Build a CLSM filled with photons in *channels* AND *micro_time_ranges*.

    Implements the user-defined detector/PIE-window selection (prompt/delay/…):
    ``CLSMImage.fill`` supports ``micro_time_ranges`` so the image is gated by
    the window's raw micro-time channels in addition to its detector channels.

    Parameters
    ----------
    tttr : tttrlib.TTTR
        Source photon data.
    channels : sequence of int
        The window's detector channels.
    micro_time_ranges : sequence of (int, int), optional
        Raw micro-time ranges; empty/None keeps the full micro-time axis.
    """
    import tttrlib

    chs = list(channels) if channels else [0]
    mtr = [tuple(int(v) for v in r) for r in (micro_time_ranges or [])]
    # Build the marker geometry unfilled (cheap), then fill exactly once with the
    # channel + micro-time selection. Filling twice (fill=True then a gated
    # refill) roughly doubled the per-window cost.
    clsm = tttrlib.CLSMImage(tttr, channels=chs, fill=False)
    clsm.fill(tttr, channels=chs, micro_time_ranges=mtr, clear=True)
    return clsm


#: Map a detector-window name to the per-pixel intensity column name expected by
#: ndxplorer's MFD equations (so 'FRET efficiency', 'Proximity ratio', ... compute
#: directly). Detector windows already encode the PIE combos (green = prompt green,
#: red = prompt red, yellow = delayed acceptor).
MFD_INTENSITY_COLUMNS = {
    "green": "S prompt green (kHz)",
    "red": "S prompt red (kHz)",
    "yellow": "S delayed yellow (kHz)",
}


def mfd_intensity_column(window_name: str) -> str:
    """Return the ndxplorer-MFD intensity column name for a detector window."""
    return MFD_INTENSITY_COLUMNS.get(str(window_name).strip().lower(), f"S {window_name} (kHz)")


def windows_from_payload(payload: dict) -> dict[str, dict]:
    """Extract detector windows from a setup payload.

    Reads the shared detector-setup payload (``payload['detectors']``) produced by
    the step-0 detector wizard. Each entry keeps ``chs`` plus the parallel /
    perpendicular channels (``ch_p``/``ch_s``, for polarization-resolved counts)
    and ``micro_time_ranges``. Returns an empty dict when no detectors are set.
    """
    dets = (payload or {}).get("detectors") or {}
    out: dict[str, dict] = {}
    for name, det in dets.items():
        if not isinstance(det, dict):
            continue
        out[str(name)] = {
            "chs": list(det.get("chs", []) or []),
            "ch_p": list(det.get("ch_p", []) or []),
            "ch_s": list(det.get("ch_s", []) or []),
            "micro_time_ranges": [tuple(r) for r in (det.get("micro_time_ranges") or [])],
        }
    return out


def shift_wrap(arr: Any, shift: float) -> np.ndarray:
    """Circularly shift a 1-D array by *shift* channels (wrap-around).

    Like the micro-time shifter (``(t + shift) % n``): the shifted-out tail wraps
    to the front rather than being zero-filled — correct for a periodic IRF. The
    fractional part is linearly interpolated, also circularly.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not shift:
        return arr.copy()
    int_shift = int(np.floor(shift))
    frac = float(shift) - int_shift
    rolled = np.roll(arr, int_shift)
    if frac:
        rolled = (1.0 - frac) * rolled + frac * np.roll(rolled, 1)
    return rolled


def prepare_irf_hist(
    vv, vh, shift_vv: float = 0.0, shift_vh: float = 0.0,
    background_vv=None, background_vh=None,
) -> dict[str, np.ndarray]:
    """Prepare **already-computed** VV/VH IRF histograms (all histogram ops, fast).

    Circularly shifts each component (wrap-around), subtracts a per-polarization
    background (explicit, else the component minimum), and normalises each to unit
    area. Kept separate from :func:`prepare_irf` so interactive shift/background
    changes never re-histogram the raw photons.
    """

    def _prep(hist, background, shift) -> np.ndarray:
        hist = np.asarray(hist, dtype=float)
        if hist.size == 0:
            return hist
        if shift:
            hist = shift_wrap(hist, shift)
        bg = float(background) if background is not None else float(hist.min())
        hist = np.clip(hist - bg, 0.0, None)
        total = hist.sum()
        return hist / total if total > 0 else hist

    return {
        "vv": _prep(vv, background_vv, shift_vv),
        "vh": _prep(vh, background_vh, shift_vh),
    }


def raw_irf_components(irf_files, ch_p, ch_s, n_channels: int | None = None) -> dict[str, np.ndarray]:
    """Return the raw (un-prepared) VV/VH IRF histograms, summed over *irf_files*."""

    def _component(channels) -> np.ndarray:
        acc: np.ndarray | None = None
        for path in irf_files or []:
            hist = micro_time_histogram(get_tttr(path), channels, n_channels=n_channels)
            acc = hist if acc is None else acc + hist
        return acc if acc is not None else np.zeros(int(n_channels or 1), dtype=float)

    return {"vv": _component(ch_p or []), "vh": _component(ch_s or [])}


def prepare_irf(
    irf_files, ch_p, ch_s, n_channels: int | None = None,
    background_vv=None, background_vh=None, shift_vv: float = 0.0, shift_vh: float = 0.0,
) -> dict[str, np.ndarray]:
    """Prepare a per-detector IRF from files: split VV/VH, shift, bg-correct, normalise.

    Convenience over :func:`raw_irf_components` + :func:`prepare_irf_hist` (mirrors
    the smMLE / anisotropy-wizard handling). Interactive callers should cache the
    raw components and call :func:`prepare_irf_hist` directly.
    """
    raw = raw_irf_components(irf_files, ch_p, ch_s, n_channels=n_channels)
    return prepare_irf_hist(
        raw["vv"], raw["vh"], shift_vv=shift_vv, shift_vh=shift_vh,
        background_vv=background_vv, background_vh=background_vh,
    )


def detector_ps_channels(det: dict) -> tuple[list[int], list[int]]:
    """Return a detector's ``(parallel, perpendicular)`` channels.

    Uses explicit ``ch_p``/``ch_s`` when present, else splits ``chs`` the same
    way the pixel-MLE does (even indices → parallel, odd → perpendicular).
    """
    ch_p = [int(x) for x in (det.get("ch_p") or [])]
    ch_s = [int(x) for x in (det.get("ch_s") or [])]
    if ch_p or ch_s:
        return ch_p, ch_s
    chs = [int(x) for x in (det.get("chs") or [])]
    if not chs:
        return [], []
    return chs[::2], (chs[1::2] if len(chs) > 1 else chs)


def total_line_durations(clsm: Any) -> tuple[np.ndarray, int]:
    """Return ``(per-line total dwell summed over frames, n_pixel)`` for a CLSM.

    Used to convert per-pixel photon counts to a count rate (kHz), matching the
    pixel-MLE's ``counts / (line_duration / n_pixel * 1000)``.

    Line dwell is a raster-geometry property (independent of the frame), so the
    first frame is sampled and scaled by ``n_frames`` — ``n_lines`` calls instead
    of ``n_frames × n_lines``, which was a hot Python loop.
    """
    n_frames, n_lines, n_pixel = clsm.shape
    durations = np.fromiter(
        (clsm.get_line_duration(0, j) for j in range(n_lines)),
        dtype=float,
        count=n_lines,
    )
    return durations * float(n_frames), n_pixel


def phasor_maps(
    clsm: Any,
    tttr: Any,
    frequency: float = -1.0,
    tttr_irf: Any = None,
    n_ph_min: int = 2,
) -> dict[str, np.ndarray]:
    """Return per-pixel phasor maps ``g``, ``s`` and ``n_photons``.

    Thin wrapper over ``tttrlib.CLSMImage.get_phasor`` with all frames pooled per
    pixel (``stack_frames=True``). If ``tttr_irf`` is given, tttrlib applies the
    reference/IRF calibration (rotation + scaling onto the universal circle).

    Parameters
    ----------
    clsm : tttrlib.CLSMImage
        A filled CLSM image.
    tttr : tttrlib.TTTR
        The dataset the image was built from.
    frequency : float
        Modulation frequency; ``-1`` auto-derives it from the TTTR header.
    tttr_irf : tttrlib.TTTR, optional
        Reference/IRF dataset for phasor calibration.
    n_ph_min : int
        Minimum photons per pixel (below → 0/0 phasor).

    Returns
    -------
    dict of numpy.ndarray
        2-D maps ``g``, ``s`` and ``n_photons``.
    """
    ph = np.asarray(
        clsm.get_phasor(tttr, tttr_irf, float(frequency), int(n_ph_min), True),
        dtype=float,
    )
    # Shape is (n_frames, n_lines, n_pixel, 2) with stack_frames -> n_frames == 1.
    if ph.ndim == 4:
        ph = ph[0]
    g = np.nan_to_num(ph[..., 0])
    s = np.nan_to_num(ph[..., 1])
    intensity = np.asarray(clsm.get_intensity(), dtype=float)
    n_photons = intensity.sum(axis=0) if intensity.ndim == 3 else intensity
    return {"g": g, "s": s, "n_photons": n_photons}


def phasor_frames(
    clsm: Any,
    tttr: Any,
    frequency: float = -1.0,
    tttr_irf: Any = None,
    n_ph_min: int = 2,
) -> dict[str, np.ndarray]:
    """Return **per-frame** phasor stacks ``g`` and ``s`` (each ``(n_frames, y, x)``).

    Like :func:`phasor_maps` but **unstacked** (``stack_frames=False``), so each
    acquisition frame yields its own phasor image — the movie source for the
    phasor tool. Display-only (frame-resolved phasors are noisier than the pooled
    map and are not written to the HDF5).
    """
    ph = np.asarray(
        clsm.get_phasor(tttr, tttr_irf, float(frequency), int(n_ph_min), False),
        dtype=float,
    )
    # (n_frames, n_lines, n_pixel, 2) with stack_frames=False.
    return {"g": np.nan_to_num(ph[..., 0]), "s": np.nan_to_num(ph[..., 1])}


def maps_to_dataframe(maps: dict[str, np.ndarray]):
    """Flatten 2-D per-pixel maps to a per-pixel-row pandas ``DataFrame``.

    Columns follow the pixel-wise-MLE convention: ``X pixel``, ``Y pixel`` plus
    one column per map. Rows are ordered row-major (``Y`` outer, ``X`` inner).

    Parameters
    ----------
    maps : dict of numpy.ndarray
        Named 2-D maps of identical shape ``(n_lines, n_pixel)``.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    shapes = {np.asarray(v).shape for v in maps.values()}
    if len(shapes) != 1:
        raise ValueError(f"maps must share one 2-D shape, got {shapes!r}")
    ny, nx = next(iter(shapes))
    yy, xx = np.indices((ny, nx))
    # Standard pixel-wise-MLE imaging layout: Y pixel = line, X pixel =
    # pixel-in-line, Pixel Number = Y * n_pixel + X.
    data = {
        "Y pixel": yy.ravel(),
        "X pixel": xx.ravel(),
        "Pixel Number": (yy * nx + xx).ravel(),
    }
    for name, arr in maps.items():
        data[name] = np.asarray(arr, dtype=float).ravel()
    return pd.DataFrame(data)


def write_imaging_hdf5(df, path: str, source: str | None = None) -> None:
    """Write a per-pixel DataFrame in the standard imaging-HDF5 format.

    Uses ``key='results'`` and a PyTables ``table`` layout with BLOSC
    compression, identical to the pixel-wise MLE output, so the file is the
    interchangeable standard imaging format (and readable by ndxplorer). When
    *source* is given, a ``meta`` table stores a back-reference to the original
    photon-data (TTTR) file so downstream tools (e.g. CLSM Draw) can trace back.
    """
    df.to_hdf(path, key="results", mode="w", complevel=9, complib="blosc", format="table")
    if source is not None:
        import pandas as pd

        pd.DataFrame([{"source_tttr": str(source)}]).to_hdf(
            path, key="meta", mode="a", complevel=9, complib="blosc", format="table"
        )


def read_imaging_source(path: str) -> str | None:
    """Return the back-referenced source-TTTR path stored in an imaging HDF5."""
    import pandas as pd

    try:
        return str(pd.read_hdf(path, key="meta")["source_tttr"].iloc[0])
    except Exception:
        return None


def add_maps_to_hdf5(path: str, maps: dict[str, np.ndarray], key: str = "results") -> list[str]:
    """Add per-pixel maps as additional columns to an imaging HDF5 in place.

    Rather than writing a separate file, the new maps are merged as extra data
    fields into the existing per-pixel ``results`` table (e.g. a pixel-wise-MLE
    lifetime map), aligned by ``X pixel`` / ``Y pixel``. Frame-collapsed maps
    (N&B, phasor) broadcast across any ``Z pixel`` / frame rows. If the file
    (or a compatible ``results`` table) does not exist, it is created from the
    maps alone.

    Parameters
    ----------
    path : str
        Target imaging HDF5 file.
    maps : dict of numpy.ndarray
        Named 2-D maps to add (e.g. ``{"N", "B", "epsilon"}`` or ``{"g", "s"}``).
    key : str
        HDF5 table key (default ``"results"``).

    Returns
    -------
    list of str
        The column names that were added/updated.
    """
    import os

    import pandas as pd

    new = maps_to_dataframe(maps)
    _keys = ("X pixel", "Y pixel", "Pixel Number")
    add_cols = [c for c in new.columns if c not in _keys]

    base = None
    if os.path.exists(path):
        try:
            base = pd.read_hdf(path, key=key)
        except Exception:
            base = None

    if base is not None and {"X pixel", "Y pixel"}.issubset(base.columns):
        # Drop any stale copies of the incoming columns, then merge by pixel.
        base = base.drop(columns=[c for c in add_cols if c in base.columns], errors="ignore")
        merged = base.merge(
            new[["X pixel", "Y pixel", *add_cols]],
            on=["X pixel", "Y pixel"],
            how="left",
        )
    else:
        merged = new

    # Preserve the back-reference to the original photon data across the rewrite.
    source = read_imaging_source(path)
    write_imaging_hdf5(merged, path, source=source)
    return add_cols
