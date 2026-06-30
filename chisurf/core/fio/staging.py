"""Staged loading of large data files from slow (network) storage.

Large TTTR files (PTU/HT3/SPC) are read by ``tttrlib`` in a single blocking
C++ call that offers no progress callback, so loading a multi-GB file from a
slow network share freezes the caller (and, in the GUI, the whole interface).

This module provides a transport-agnostic, **Qt-free** building block that
gives back genuine progress *and* transfer speed for that case:

1. Probe the source throughput by timing a small head read.
2. If the source is *slow* (and large enough to matter), stream-copy it to a
   local temporary file with our own chunked read loop -- which is where the
   ``progress_cb`` (bytes/percent/MB-s/ETA) information comes from -- and hand
   the local copy to the reader. The subsequent ``tttrlib`` parse then runs
   against fast local disk.
3. If the source is *fast* (local SSD, fast NAS) the probe bytes are discarded
   and the original path is returned unchanged -- no needless copy.

The staged copy is **ephemeral**: :func:`staged_source` deletes it once the
caller is done. Because there is no Qt dependency here, this is usable from the
headless server/CLI as well as the GUI; the GUI layer
(``chisurf.gui.widgets.staged_loading``) only adds a progress dialog on top.

Examples
--------
>>> from chisurf.core.fio import staging
>>> with staging.staged_source(path) as local:  # doctest: +SKIP
...     tttr = tttrlib.TTTR(str(local))

Readers that just want a ``tttrlib.TTTR`` use :func:`open_tttr`, which stages
on slow storage, parses, and deletes the temp in one call.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

__all__ = [
    "StagingCancelled",
    "ProgressCallback",
    "CancelCallback",
    "stage_path_if_slow",
    "staged_source",
    "open_tttr",
    "format_rate",
]

# --- Tunables ---------------------------------------------------------------
# Defaults may be overridden via the ``data_loading`` section of
# ``chisurf.settings.cs_settings`` (see :func:`_settings`), which is editable
# through the "Data loading" settings AutoForm. They are intentionally
# conservative: the only cost of a wrong "slow" guess is one extra local copy;
# the cost of a wrong "fast" guess is a frozen-feeling load with no progress.

#: Files smaller than this are loaded directly -- staging overhead is not worth
#: it and a progress bar on a sub-second copy only flickers.
DEFAULT_MIN_SIZE = 32 * 1024 * 1024  # 32 MiB

#: Sources slower than this (measured over the probe) are staged locally.
DEFAULT_THRESHOLD_MBPS = 40.0

#: Chunk size for both the probe and the copy loop.
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB

#: How much of the head to read before deciding slow-vs-fast...
DEFAULT_PROBE_BYTES = 8 * 1024 * 1024  # 8 MiB
#: ...but stop probing early once this much wall-clock has elapsed, so we never
#: block for long on a very slow link just to make the decision.
DEFAULT_PROBE_MIN_SECONDS = 0.3
#: ...and require at least this many bytes before trusting the measurement.
DEFAULT_PROBE_MIN_BYTES = 1 * 1024 * 1024  # 1 MiB

#: Minimum wall-clock between successive ``progress_cb`` invocations, to avoid
#: flooding a GUI marshal queue on a fast link.
DEFAULT_PROGRESS_INTERVAL = 0.1


#: ``progress_cb(bytes_done, total_bytes, mbps, eta_seconds)``. ``eta_seconds``
#: is ``None`` until a rate can be estimated.
ProgressCallback = Callable[[int, int, float, float | None], None]
#: ``cancel_cb() -> bool``; return ``True`` to abort the copy.
CancelCallback = Callable[[], bool]


class StagingCancelled(Exception):
    """Raised by :func:`stage_path_if_slow` when ``cancel_cb`` requests abort."""


#: Default values for the ``data_loading`` settings section. These seed both
#: the runtime fallbacks (:func:`_settings`) and the AutoForm JSON view-spec
#: (``chisurf/gui/widgets/staged_loading_view.json``); keep the two in sync.
DEFAULTS = {
    "enabled": True,
    "min_size": DEFAULT_MIN_SIZE,
    "threshold_mbps": DEFAULT_THRESHOLD_MBPS,
    "chunk_bytes": DEFAULT_CHUNK_BYTES,
    "probe_bytes": DEFAULT_PROBE_BYTES,
    "probe_min_seconds": DEFAULT_PROBE_MIN_SECONDS,
    "probe_min_bytes": DEFAULT_PROBE_MIN_BYTES,
    "progress_interval": DEFAULT_PROGRESS_INTERVAL,
}


def _settings() -> dict:
    """Return the ``data_loading`` settings section merged over :data:`DEFAULTS`."""
    cfg = dict(DEFAULTS)
    try:
        import chisurf.settings as settings

        section = settings.cs_settings.get("data_loading")
        if isinstance(section, dict):
            cfg.update(section)
    except Exception:
        pass
    return cfg


def _cache_dir() -> Path:
    """Directory used for ephemeral staging copies."""
    try:
        import chisurf.settings as settings

        base = Path(settings._chisurf_user_cache_dir)
    except Exception:
        base = Path(tempfile.gettempdir()) / "chisurf"
    staging = base / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def format_rate(mbps: float) -> str:
    """Format a transfer rate in MB/s (or GB/s for fast links)."""
    if mbps >= 1000.0:
        return f"{mbps / 1000.0:.2f} GB/s"
    return f"{mbps:.1f} MB/s"


def _emit(
    cb: ProgressCallback | None,
    done: int,
    total: int,
    t0: float,
    state: dict,
    *,
    force: bool = False,
) -> None:
    """Throttled progress emission with rate/ETA derived from ``t0``."""
    if cb is None:
        return
    now = perf_counter()
    if not force and (now - state["last"]) < state["interval"]:
        return
    state["last"] = now
    elapsed = now - t0
    rate_bps = (done / elapsed) if elapsed > 0 else 0.0
    mbps = rate_bps / 1e6
    eta = ((total - done) / rate_bps) if rate_bps > 0 else None
    cb(done, total, mbps, eta)


def stage_path_if_slow(
    src,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    min_size: int | None = None,
    threshold_mbps: float | None = None,
    chunk_bytes: int | None = None,
    probe_bytes: int | None = None,
) -> tuple[Path, bool]:
    """Copy *src* to a fast local temp file when it lives on slow storage.

    The source is opened once. A small head is read and timed; if the measured
    throughput is at or above ``threshold_mbps`` (or the file is below
    ``min_size``, or the whole file fit in the probe) the original path is
    returned unchanged. Otherwise the already-read probe bytes plus the rest of
    the file are streamed to a temp file in the chisurf cache dir, invoking
    ``progress_cb`` throughout, and that temp path is returned.

    Parameters
    ----------
    src : str or pathlib.Path
        Source file to (maybe) stage.
    progress_cb : callable, optional
        Called as ``progress_cb(bytes_done, total_bytes, mbps, eta_seconds)``
        during the copy. Not called when the file is not staged.
    cancel_cb : callable, optional
        Polled between chunks; if it returns ``True`` the partial copy is
        removed and :class:`StagingCancelled` is raised.
    min_size, threshold_mbps, chunk_bytes, probe_bytes
        Override the module defaults (themselves overridable via settings).

    Returns
    -------
    (pathlib.Path, bool)
        ``(path_to_open, was_staged)``. When ``was_staged`` is ``True`` the
        caller owns the returned temp file and must delete it (use
        :func:`staged_source` to do this automatically).
    """
    cfg = _settings()
    min_size = int(cfg["min_size"] if min_size is None else min_size)
    threshold_mbps = float(cfg["threshold_mbps"] if threshold_mbps is None else threshold_mbps)
    chunk_bytes = int(cfg["chunk_bytes"] if chunk_bytes is None else chunk_bytes)
    probe_bytes = int(cfg["probe_bytes"] if probe_bytes is None else probe_bytes)
    probe_min_seconds = float(cfg["probe_min_seconds"])
    probe_min_bytes = int(cfg["probe_min_bytes"])
    interval = float(cfg["progress_interval"])

    src = Path(src)
    # Staging globally disabled, or file too small for staging to be worthwhile.
    if not cfg.get("enabled", True):
        return src, False
    total = src.stat().st_size
    if total < min_size:
        return src, False

    state = {"last": 0.0, "interval": interval}

    fh = open(src, "rb")
    try:
        # --- Probe: read the head and time it -------------------------------
        probe = bytearray()
        t0 = perf_counter()
        while len(probe) < probe_bytes:
            if cancel_cb is not None and cancel_cb():
                raise StagingCancelled()
            chunk = fh.read(min(chunk_bytes, probe_bytes - len(probe)))
            if not chunk:
                break  # whole file smaller than probe window
            probe += chunk
            elapsed = perf_counter() - t0
            if elapsed >= probe_min_seconds and len(probe) >= probe_min_bytes:
                break

        elapsed = perf_counter() - t0
        mbps = ((len(probe) / 1e6) / elapsed) if elapsed > 0 else float("inf")

        # Fast enough, or the probe already drained the file -> don't stage.
        if mbps >= threshold_mbps or len(probe) >= total:
            return src, False

        # --- Slow: stream-copy probe bytes + remainder to a temp file -------
        staging_dir = Path(tempfile.mkdtemp(prefix="stage-", dir=_cache_dir()))
        # Preserve the original filename so tttrlib's container-type detection
        # (which keys off the extension/name) behaves identically.
        dst = staging_dir / src.name
        try:
            done = 0
            with open(dst, "wb") as out:
                out.write(probe)
                done = len(probe)
                _emit(progress_cb, done, total, t0, state, force=True)
                while True:
                    if cancel_cb is not None and cancel_cb():
                        raise StagingCancelled()
                    chunk = fh.read(chunk_bytes)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    _emit(progress_cb, done, total, t0, state)
            _emit(progress_cb, done, total, t0, state, force=True)
            return dst, True
        except BaseException:
            # Cancel or I/O error: drop the partial copy before propagating.
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    finally:
        fh.close()


@contextlib.contextmanager
def staged_source(
    src,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    **kwargs,
):
    """Context manager yielding a local path, deleting any staged temp on exit.

    >>> with staged_source(path) as local:        # doctest: +SKIP
    ...     tttr = tttrlib.TTTR(str(local))
    """
    local, staged = stage_path_if_slow(src, progress_cb=progress_cb, cancel_cb=cancel_cb, **kwargs)
    try:
        yield local
    finally:
        if staged:
            shutil.rmtree(local.parent, ignore_errors=True)


def open_tttr(
    src,
    routine=None,
    *,
    progress_cb: ProgressCallback | None = None,
    cancel_cb: CancelCallback | None = None,
    **stage_kwargs,
):
    """Build a :class:`tttrlib.TTTR`, staging *src* locally first when slow.

    Drop-in replacement for ``tttrlib.TTTR(path[, routine])`` that adds the
    slow-storage staging + progress behaviour. The staged temp copy (if any) is
    deleted once parsing finishes -- ``tttrlib`` has the events in memory by
    then, so the on-disk copy is no longer needed.

    Parameters
    ----------
    src : str or pathlib.Path
        File to read.
    routine : str or int, optional
        ``tttrlib`` container/reading-routine argument. ``None``/empty means
        auto-detect from the filename (which staging preserves).
    progress_cb, cancel_cb
        Forwarded to :func:`stage_path_if_slow`.
    **stage_kwargs
        Forwarded to :func:`stage_path_if_slow` (e.g. ``threshold_mbps``).
    """
    import tttrlib

    with staged_source(src, progress_cb=progress_cb, cancel_cb=cancel_cb, **stage_kwargs) as local:
        if routine is None or routine == "":
            return tttrlib.TTTR(str(local))
        return tttrlib.TTTR(str(local), routine)
