"""Qt-free view-model for the Intensity imaging tool.

Owns **all** per-pixel intensity/count columns in the standard imaging HDF5.
For every detector window (step-0 Setup) it writes, matching the names the
pixel-MLE used to emit (so nothing was lost when the MLE became fit-only):

- ``N{c}-p-all`` / ``N{c}-s-all`` / ``N{c}-all`` — parallel / perpendicular /
  total photon counts (``c`` = detector initial),
- ``<detector> Count Rate (KHz)`` and the ndxplorer-MFD name (e.g.
  ``S prompt green (kHz)``) — per-pixel count rate,
- a global ``Number of Photons`` (sum over windows),

plus a back-reference to the source photon data. N&B / phasor / MLE then enrich
this file with their own columns.
"""

from __future__ import annotations

import pathlib

import numpy as np

from chisurf.plugins.microscopy.imaging_common.base import ImagingMapViewModel

_VIEW_JSON = pathlib.Path(__file__).parent / "intensity.view.json"


class IntensityViewModel(ImagingMapViewModel):
    """State + logic for the interactive Intensity imaging tool (no Qt)."""

    HDF5_ACTION_LABEL = "💾 Create imaging HDF5"

    def __init__(self) -> None:
        super().__init__(_VIEW_JSON)

    def compute(self, progress=None) -> bool:
        """Compute per-window intensity/count columns (MLE-compatible names).

        Qt-free (safe off the UI thread); the base ``run``/GUI handle notify.
        """
        try:
            from chisurf.core.fluorescence.imaging import compute_windows, mfd_intensity_column

            # Detector windows are computed in parallel across processes.
            results = compute_windows(
                self.filename, self._windows(), "intensity", progress=progress
            )
            by_window: dict[str, dict] = {}
            columns: dict[str, np.ndarray] = {}
            shape = None
            total_photons = None
            dwell = None
            for win, r in results.items():
                n_par, n_perp = r["n_par"], r["n_perp"]
                n_all = n_par + n_perp
                shape = n_all.shape
                # Line dwell is geometry-only → use the first window's durations.
                if dwell is None:
                    dwell = (r["durations"] / max(r["n_pixel"], 1))[:, None]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = np.where(dwell > 0, n_all / (dwell * 1000.0), 0.0)
                rate = np.nan_to_num(rate)
                # Per-detector background subtraction (kHz) from the IRF & BG step.
                bg = float(r.get("bg", 0.0))
                if bg:
                    rate = np.clip(rate - bg, 0.0, None)

                c = (win or "d").strip().lower()[:1] or "d"
                columns[f"N{c}-p-all"] = n_par
                columns[f"N{c}-s-all"] = n_perp
                columns[f"N{c}-all"] = n_all
                columns[f"{win} Count Rate (KHz)"] = rate
                columns[mfd_intensity_column(win)] = rate
                by_window[win] = {"intensity": n_all, "count_rate": rate, "frames": r.get("frames")}
                total_photons = n_all if total_photons is None else total_photons + n_all
            if total_photons is not None:
                columns["Number of Photons"] = total_photons
        except Exception as exc:
            self.results_text = f"Computation failed: {exc}"
            return False
        self._last_signature = self._signature()
        self._by_window = by_window
        self._columns = columns
        if self.display_window not in by_window and by_window:
            self.display_window = next(iter(by_window))
        if shape is not None:
            self.results_text = self._summary(*shape)
        # Warm the raw-frame movie stack on the bg compute thread (see base).
        self._warm_movie_cache()
        return True

    def count_rate_map(self):
        """Return the count-rate (kHz) map of the displayed window."""
        return self._disp("count_rate")

    def _write_hdf5(self, path: str) -> list[str]:
        """Create a fresh standard imaging HDF5 with the source back-reference."""
        from chisurf.core.fluorescence.imaging import maps_to_dataframe, write_imaging_hdf5

        write_imaging_hdf5(maps_to_dataframe(self._columns), path, source=self.filename or None)
        return list(self._columns.keys())
