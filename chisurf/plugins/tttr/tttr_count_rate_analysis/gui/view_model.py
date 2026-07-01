"""Qt-free view-model backing the Count Rate Analysis tool.

:class:`CountRateViewModel` holds the dropped TTTR file list and the per-channel
count-rate results, and computes count rates through :mod:`tttrlib`. The channel
definition comes from the (Qt) ``DetectorWizardPage`` via an injected
``channels_provider`` callable, keeping this model free of Qt so it is unit-
testable headlessly. The GUI (``gui.tool`` + ``gui.sections``) owns Qt concerns.

Mirrors :class:`chisurf.plugins.tttr.ptu_alex_creator.gui.view_model.AlexViewModel`.
"""

from __future__ import annotations

import logging
import os
import pathlib
from collections.abc import Callable

import numpy as np
import tttrlib

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "count_rate.view.json"


class CountRateViewModel:
    """State + logic for the Count Rate Analysis tool (no Qt)."""

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored ``count_rate.view.json``."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    def __init__(self) -> None:
        self.files: list[str] = []
        #: Callable returning the DetectorWizardPage channels map (set by the GUI).
        self.channels_provider: Callable[[], dict] | None = None

        # Results (populated by :meth:`compute`).
        self._per_file: dict[str, dict[str, float]] = {}
        self._per_file_photons: dict[str, dict[str, int]] = {}
        self._meas_times: dict[str, float] = {}
        self._channel_order: list[str] = []

        self._observers: list[Callable[[str], None]] = []

    # ── observer hook ──────────────────────────────────────────────────
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("count-rate observer failed", exc_info=True)

    def update(self) -> None:
        """AutoForm hook: the ``path_list`` section wrote ``files`` — refresh views."""
        self.notify("files")

    # ── file list ──────────────────────────────────────────────────────
    def add_files(self, paths: list[str]) -> None:
        """Add unique TTTR files, sorted lexically by basename."""
        added = False
        for p in paths:
            if p and p not in self.files:
                self.files.append(p)
                added = True
        if added:
            self.files.sort(key=lambda p: os.path.basename(p).lower())
            self.notify("files")

    def clear(self) -> None:
        """Drop all files and results."""
        self.files = []
        self._per_file = {}
        self._per_file_photons = {}
        self._meas_times = {}
        self._channel_order = []
        self.notify("files")

    # ── compute ────────────────────────────────────────────────────────
    def can_compute(self) -> str | None:
        """Return ``None`` when a computation can run, else a reason string."""
        if not self.files:
            return "Please load TTTR files first."
        channels = self._channels()
        if not channels:
            return "Please define channels in the detector wizard."
        return None

    def _channels(self) -> dict:
        try:
            return self.channels_provider() if callable(self.channels_provider) else {}
        except Exception:
            logger.warning("count-rate: channels_provider failed", exc_info=True)
            return {}

    def compute(self) -> None:
        """Compute per-file, per-channel count rates for every loaded file."""
        reason = self.can_compute()
        if reason is not None:
            raise ValueError(reason)
        channels = self._channels()

        per_file: dict[str, dict[str, float]] = {}
        per_file_photons: dict[str, dict[str, int]] = {}
        meas_times: dict[str, float] = {}

        for path in self.files:
            tttr = tttrlib.TTTR(path)
            mtr = tttr.header.macro_time_resolution
            meas_time = float(tttr.macro_times[-1] * mtr) if len(tttr.macro_times) else 0.0
            meas_times[path] = meas_time

            rates: dict[str, float] = {}
            nphot: dict[str, int] = {}
            for cname, infos in channels.items():
                crs: list[float] = []
                nps: list[int] = []
                for info in infos:
                    tf = tttr.get_tttr_by_channel(info["detector_chs"])
                    tf = self._apply_range(tf, info.get("micro_time_range"))
                    tf = self._apply_range(tf, info.get("window_range"))
                    n = len(tf.macro_times)
                    nps.append(n)
                    if meas_time > 0:
                        crs.append(n / meas_time)
                rates[cname] = float(np.mean(crs)) if crs else 0.0
                nphot[cname] = int(sum(nps))
            per_file[path] = rates
            per_file_photons[path] = nphot

        self._per_file = per_file
        self._per_file_photons = per_file_photons
        self._meas_times = meas_times
        self._channel_order = list(channels.keys())
        self.notify("computed")

    @staticmethod
    def _apply_range(tf, rng):
        """Select the sub-set of *tf* whose micro-times fall within *rng*."""
        if not rng:
            return tf
        micro = tf.micro_times
        mask = (micro >= rng[0]) & (micro <= rng[1])
        return tf.get_tttr_by_selection(np.where(mask)[0].astype(np.int32))

    # ── AutoForm accessors ─────────────────────────────────────────────
    def results_rows(self) -> list[dict]:
        """Per-channel summary rows (mean/std kHz, total photons, total time)."""
        if not self._per_file:
            return []
        rows = []
        total_time = float(np.sum(list(self._meas_times.values())))
        for ch in self._channel_order:
            rates = [fr.get(ch, 0.0) for fr in self._per_file.values()]
            photons = [fp.get(ch, 0) for fp in self._per_file_photons.values()]
            rows.append(
                {
                    "channel": ch,
                    "mean_khz": float(np.mean(rates)) / 1000.0,
                    "std_khz": float(np.std(rates)) / 1000.0,
                    "photons": int(np.sum(photons)),
                    "time_s": total_time,
                }
            )
        return rows

    def count_rate_series(self) -> list[dict]:
        """Per-channel count-rate (kHz) vs file-index series for the plot."""
        if not self._per_file:
            return []
        colors = ["#4488ff", "#ff5555", "#55cc55", "#22cccc", "#cc55cc", "#cccc22"]
        x = np.arange(len(self.files))
        series = []
        for i, ch in enumerate(self._channel_order):
            y = [self._per_file.get(f, {}).get(ch, 0.0) / 1000.0 for f in self.files]
            series.append(
                {
                    "x": x,
                    "y": np.array(y),
                    "name": ch,
                    "color": colors[i % len(colors)],
                    "width": 2,
                    "symbol": "o",
                    "symbol_size": 7,
                }
            )
        return series

    # ── export ─────────────────────────────────────────────────────────
    def can_save(self) -> str | None:
        """Return ``None`` when results exist to save, else a reason string."""
        return None if self._per_file else "There is no data to save."

    def save_table(self, path: str) -> None:
        """Write the per-channel results table to *path* as tab-separated text."""
        rows = self.results_rows()
        if not rows:
            raise ValueError("There is no data to save.")
        headers = [
            "Channel",
            "Mean Count Rate (kHz)",
            "Std Count Rate (kHz)",
            "#Photons",
            "Measurement Time (s)",
        ]
        with open(path, "w") as fh:
            fh.write("\t".join(headers) + "\n")
            for r in rows:
                fh.write(
                    f"{r['channel']}\t{r['mean_khz']:.2f}\t{r['std_khz']:.2f}\t"
                    f"{r['photons']:.0f}\t{r['time_s']:.3f}\n"
                )


__all__ = ["CountRateViewModel"]
