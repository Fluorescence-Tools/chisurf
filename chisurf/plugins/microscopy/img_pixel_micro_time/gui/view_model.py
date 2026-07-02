"""Qt-free view-model backing the mean-micro-time imaging tool.

Computes the per-pixel mean micro-time (photon-weighted arrival time) for
**every** detector window defined in the step-0 setup, and adds it to the
standard imaging HDF5.
"""

from __future__ import annotations

import pathlib

from chisurf.plugins.microscopy.imaging_common.base import ImagingMapViewModel

_VIEW_JSON = pathlib.Path(__file__).parent / "micro_time.view.json"


class MicroTimeViewModel(ImagingMapViewModel):
    """State + logic for the interactive mean-micro-time imaging tool (no Qt)."""

    HDF5_ACTION_LABEL = "➕ Add mean micro-time to HDF5"
    WINDOW_KIND = "mean_micro_time"

    def __init__(self) -> None:
        super().__init__(_VIEW_JSON)
        #: Minimum photons per pixel (below → 0) for the mean-micro-time estimate.
        self.n_ph_min: int = 2

    def _window_params(self) -> dict:
        """Pass the photon threshold to the worker (resolution auto-derived)."""
        return {"n_ph_min": int(self.n_ph_min), "microtime_resolution": -1.0}

    def _extra_signature(self) -> tuple:
        return (int(self.n_ph_min),)

    # ── image accessors (view.json `image` sections; displayed window) ──
    def mean_micro_time_map(self):
        """Return the mean-micro-time (ns) map of the displayed window."""
        return self._disp("mean_micro_time")

    def mean_micro_time_frames(self):
        """Return the per-frame mean-micro-time stack ``(n_frames, y, x)`` (ns).

        The movie source for the ``image`` ``movie`` control: the same
        ``get_mean_micro_time`` estimate as :meth:`mean_micro_time_map` but
        **unstacked** (``stack_frames=False``), so each acquisition frame is a
        separate mean-micro-time image. Display-only (not written to the HDF5).
        Prefers the worker-process stack (stashed under ``mt_frames``) so the movie
        never rebuilds a CLSM on the UI thread; falls back to a lazy build.
        """
        import numpy as np

        stashed = self._disp("mt_frames")
        if stashed is not None:
            return stashed
        if not self.filename:
            return None
        win = self._windows().get(self.display_window)
        if win is None:
            return None
        chs = list(win.get("chs", [0]) or [0])
        mtr = list(win.get("micro_time_ranges") or [])
        sig = (self.filename, self.display_window, tuple(chs),
               tuple(tuple(r) for r in mtr), int(self.n_ph_min))

        def build():
            from chisurf.core.fluorescence.imaging import cached_clsm, get_tttr

            tttr = get_tttr(self.filename)
            clsm = cached_clsm(self.filename, chs, mtr)
            micro_res = float(getattr(tttr.header, "micro_time_resolution", 0.0) or 0.0)
            res_ns = micro_res * 1e9 if micro_res > 0.0 else -1.0
            return np.nan_to_num(np.asarray(
                clsm.get_mean_micro_time(tttr, res_ns, int(self.n_ph_min), False), dtype=float,
            ))

        return self._cached_stack("mean_micro_time_frames", sig, build)

    def _warm_movie_cache(self) -> None:
        """Warm the raw-frame + mean-micro-time movie stacks (bg compute thread)."""
        super()._warm_movie_cache()
        self.mean_micro_time_frames()
