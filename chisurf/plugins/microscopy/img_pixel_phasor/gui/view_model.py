"""Qt-free view-model backing the per-pixel phasor-FLIM imaging tool.

Computes phasor (g, s) per pixel for **every** detector window (via tttrlib's
built-in ``get_phasor``, raw + optional IRF reference) and adds them to the
standard imaging HDF5.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from chisurf.plugins.microscopy.imaging_common.base import ImagingMapViewModel

_VIEW_JSON = pathlib.Path(__file__).parent / "phasor.view.json"


class PhasorImgViewModel(ImagingMapViewModel):
    """State + logic for the interactive phasor-FLIM imaging tool (no Qt)."""

    HDF5_ACTION_LABEL = "➕ Add phasor to HDF5"
    WINDOW_KIND = "phasor"
    #: Phasor-plot extent (data coords). Widened past the universal circle
    #: (g∈[0,1], s∈[0,0.5]) so noisy pixels near the edges are not clipped.
    PHASOR_G_RANGE = (-0.1, 1.1)
    PHASOR_S_RANGE = (-0.05, 0.7)

    def __init__(self) -> None:
        super().__init__(_VIEW_JSON)
        # phasor-specific settings (bound by the view.json `value`s)
        self.n_ph_min: int = 3
        self.frequency: float = -1.0
        self.colormap = "viridis"

    def _extra_signature(self) -> tuple:
        """Phasor settings that change the result (for recompute dedup)."""
        return (int(self.n_ph_min), float(self.frequency))

    def _window_params(self) -> dict:
        """Return phasor worker params (IRF comes per-detector from the IRF & BG step)."""
        return {"frequency": float(self.frequency), "n_ph_min": int(self.n_ph_min)}

    # ── image accessors ──
    def g_map(self):
        """Return the phasor g map of the displayed window."""
        return self._disp("g")

    def s_map(self):
        """Return the phasor s map of the displayed window."""
        return self._disp("s")

    # ── per-frame movie accessors (unstacked phasor) ──
    def _phasor_frames(self):
        """Return per-frame phasor ``g``/``s`` stacks for the window.

        Prefers the stacks produced by the compute **worker process** (stashed on
        ``_by_window`` under ``g_frames``/``s_frames``) so the movie never runs
        ``get_phasor`` on the UI thread; falls back to a lazily-built, memoized
        CLSM call for standalone use with no compute yet.
        """
        if not self.filename or not self._by_window:
            # Gate on a completed compute so a pre-Run refresh never triggers a
            # synchronous CLSM fill / get_phasor on the UI thread (warmed in bg).
            return None
        stashed = self._by_window.get(self.display_window) or {}
        if stashed.get("g_frames") is not None and stashed.get("s_frames") is not None:
            return {"g": stashed["g_frames"], "s": stashed["s_frames"]}
        win = self._windows().get(self.display_window)
        if win is None:
            return None
        irf_files = list(win.get("irf") or [])
        key = (
            self.filename, self.display_window, float(self.frequency),
            int(self.n_ph_min), tuple(irf_files),
        )
        if getattr(self, "_pf_key", None) == key and getattr(self, "_pf_cache", None) is not None:
            return self._pf_cache
        try:
            from chisurf.core.fluorescence.imaging import (
                cached_clsm,
                get_tttr,
                phasor_frames,
            )

            tttr = get_tttr(self.filename)
            clsm = cached_clsm(
                self.filename,
                list(win.get("chs", [0]) or [0]),
                list(win.get("micro_time_ranges") or []),
            )
            tttr_irf = get_tttr(irf_files[0]) if irf_files else None
            result = phasor_frames(
                clsm, tttr, frequency=float(self.frequency),
                tttr_irf=tttr_irf, n_ph_min=int(self.n_ph_min),
            )
        except Exception:
            return None
        self._pf_key, self._pf_cache = key, result
        return result

    def g_frames(self):
        """Return the per-frame phasor-g stack ``(n_frames, y, x)`` (movie source)."""
        frames = self._phasor_frames()
        return None if frames is None else frames["g"]

    def s_frames(self):
        """Return the per-frame phasor-s stack ``(n_frames, y, x)`` (movie source)."""
        frames = self._phasor_frames()
        return None if frames is None else frames["s"]

    def phasor_histogram_frames(self, bins: int = 160) -> Any:
        """Return a per-frame stack of phasor density histograms ``(n_frames, bins, bins)``.

        The movie source for the phasor *plot*: one ``log(1+count)`` density per
        acquisition frame (axis 1 = ``g``, axis 2 = ``s``), built from the same
        unstacked per-frame phasors as :meth:`g_frames`. Discriminated pixels come
        back as ``(0, 0)`` and are excluded.
        """
        win = self._windows().get(self.display_window) or {}
        sig = (
            self.filename, self.display_window, float(self.frequency),
            int(self.n_ph_min), tuple(win.get("irf") or []), int(bins),
        )

        def build():
            frames = self._phasor_frames()
            if frames is None:
                return None
            g_st, s_st = frames["g"], frames["s"]
            n = int(g_st.shape[0])
            out = np.empty((n, bins, bins), dtype=float)
            rng = [list(self.PHASOR_G_RANGE), list(self.PHASOR_S_RANGE)]
            for f in range(n):
                g, s = g_st[f], s_st[f]
                valid = np.isfinite(g) & np.isfinite(s) & ~((g == 0.0) & (s == 0.0))
                hist, _, _ = np.histogram2d(g[valid].ravel(), s[valid].ravel(), bins=bins, range=rng)
                out[f] = np.log1p(hist)
            return out

        return self._cached_stack("phasor_histogram_frames", sig, build)

    def _warm_movie_cache(self) -> None:
        """Warm the raw-frame, per-frame g/s and phasor-plot movie stacks (bg thread)."""
        super()._warm_movie_cache()
        self.g_frames()
        self.s_frames()
        self.phasor_histogram_frames()

    def phasor_histogram_map(self, bins: int = 160) -> Any:
        """Return a 2-D density histogram of the displayed window's (g, s) cloud.

        Axis 0 is ``g``, axis 1 is ``s`` (pyqtgraph ``ImageItem`` is column-major,
        so this renders with g horizontal / s vertical — no rotation), values are
        ``log(1+count)``. A 2-D histogram reads far better than a scatter for the
        dense per-pixel phasor cloud; the phasor section overlays the universal
        semicircle and calibrates the axes.
        """
        g, s, n = self._disp("g"), self._disp("s"), self._disp("n_photons")
        if g is None or s is None:
            return None
        mask = (n > 0) if n is not None else np.ones_like(g, dtype=bool)
        gg, ss = g[mask].ravel(), s[mask].ravel()
        valid = np.isfinite(gg) & np.isfinite(ss)
        hist, _, _ = np.histogram2d(
            gg[valid], ss[valid], bins=bins,
            range=[list(self.PHASOR_G_RANGE), list(self.PHASOR_S_RANGE)],
        )
        return np.log1p(hist)
