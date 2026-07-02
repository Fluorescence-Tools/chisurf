"""Qt-free view-model for the IRF & BG calibration step.

Per detector it holds an **IRF file**, a **convolution range** (micro-time
start/stop for the fit) and a **background** (kHz), edited against a live decay /
IRF plot (see the ``decay_conv`` AutoForm section). The calibration is published
to the imaging coordinator and transferred to Phasor + pixel-wise MLE.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable

logger = logging.getLogger(__name__)

_VIEW_JSON = pathlib.Path(__file__).parent / "calibration.view.json"


def _blank() -> dict:
    return {
        "irf": [], "bg_vv": 0.0, "bg_vh": 0.0, "shift_vv": 0.0, "shift_vh": 0.0,
        # Convolution/fit window, the (separate) IRF window, and the region the
        # background is estimated from.
        "conv_start": 0, "conv_stop": 0, "irf_start": 0, "irf_stop": 0,
        "bg_start": 0, "bg_stop": 0,
    }


class CalibrationViewModel:
    """State + logic for the interactive IRF & BG step (no Qt)."""

    def __init__(self) -> None:
        self.detectors: dict[str, dict] = {}
        #: Per-detector calibration: {det: {irf, bg, conv_start, conv_stop}}.
        self.calibration: dict[str, dict] = {}
        self.display_detector: str = ""
        self.filename: str = ""
        self.publish: Callable[[dict], None] | None = None
        self.status_text: str = ""
        self._observers: list[Callable[[str], None]] = []
        #: Cached histograms {(file, det, irf_files): {...}} so interactive
        #: shift/background/region changes never re-histogram the raw photons.
        self._hist_cache: dict = {}

    def view_spec(self):
        """Resolve AutoForm's view spec from the authored view.json."""
        from chisurf.core.dataspec import load_view_spec

        return load_view_spec(_VIEW_JSON)

    # ── observer hook ──
    def add_observer(self, cb: Callable[[str], None]) -> None:
        """Register *cb* to be called with an event name on every change."""
        self._observers.append(cb)

    def notify(self, event: str = "changed") -> None:
        """Notify observers that state changed."""
        for cb in list(self._observers):
            try:
                cb(event)
            except Exception:
                logger.debug("calibration observer failed", exc_info=True)

    # ── shared setup / pipeline ──
    def apply_setup_settings(self, payload: dict) -> None:
        """Sync detector rows from the shared detector definition."""
        try:
            from chisurf.core.fluorescence.imaging import windows_from_payload

            dets = windows_from_payload(payload)
        except Exception:
            dets = {}
        if not dets:
            return
        self.detectors = dets
        for name in dets:
            self.calibration.setdefault(name, _blank())
        for name in list(self.calibration):
            if name not in dets:
                self.calibration.pop(name, None)
        if self.display_detector not in dets:
            self.display_detector = next(iter(dets))
        self.notify("setup")

    def apply_pipeline_context(self, payload: dict) -> None:
        """Adopt the current source TTTR so the decay/IRF plot can be drawn."""
        source = (payload or {}).get("source")
        if source and str(source) != self.filename:
            self.filename = str(source)
            self._hist_cache.clear()
            self.notify("changed")

    # ── AutoForm accessors ──
    def window_names(self) -> list[str]:
        """Detector names (the selectable detectors)."""
        return list(self.detectors.keys())

    def refresh_display(self, *_args) -> None:
        """Re-render for the selected detector / setting."""
        self.notify("run")

    def _cur(self) -> dict:
        if not self.display_detector:
            return _blank()
        return self.calibration.setdefault(self.display_detector, _blank())

    # selected-detector fields (AutoForm binds sections to these)
    @property
    def sel_irf_files(self) -> list:
        """IRF file list of the selected detector (bound to the `path_list`)."""
        irf = self._cur().get("irf")
        return list(irf) if isinstance(irf, list) else ([irf] if irf else [])

    @sel_irf_files.setter
    def sel_irf_files(self, value) -> None:
        self._cur()["irf"] = [str(v) for v in (value or [])]
        self.notify("run")  # redraw the decay with the new IRF list

    @property
    def sel_bg_vv(self) -> float:
        """Parallel (VV) background of the selected detector (0 = auto baseline)."""
        return float(self._cur().get("bg_vv", 0.0) or 0.0)

    @sel_bg_vv.setter
    def sel_bg_vv(self, value) -> None:
        self._cur()["bg_vv"] = float(value or 0.0)
        self.notify("run")

    @property
    def sel_bg_vh(self) -> float:
        """Perpendicular (VH) background of the selected detector (0 = auto baseline)."""
        return float(self._cur().get("bg_vh", 0.0) or 0.0)

    @sel_bg_vh.setter
    def sel_bg_vh(self, value) -> None:
        self._cur()["bg_vh"] = float(value or 0.0)
        self.notify("run")

    @property
    def sel_shift_vv(self) -> float:
        """Parallel (VV) IRF shift in channels (wrap-around)."""
        return float(self._cur().get("shift_vv", 0.0) or 0.0)

    @sel_shift_vv.setter
    def sel_shift_vv(self, value) -> None:
        self._cur()["shift_vv"] = float(value or 0.0)
        self.notify("run")

    @property
    def sel_shift_vh(self) -> float:
        """Perpendicular (VH) IRF shift in channels (wrap-around)."""
        return float(self._cur().get("shift_vh", 0.0) or 0.0)

    @sel_shift_vh.setter
    def sel_shift_vh(self, value) -> None:
        self._cur()["shift_vh"] = float(value or 0.0)
        self.notify("run")

    @property
    def sel_conv_start(self) -> int:
        """Convolution-range start (micro-time channel) of the selected detector."""
        return int(self._cur().get("conv_start", 0) or 0)

    @sel_conv_start.setter
    def sel_conv_start(self, value) -> None:
        self._cur()["conv_start"] = int(value or 0)

    @property
    def sel_conv_stop(self) -> int:
        """Convolution-range stop (micro-time channel) of the selected detector."""
        return int(self._cur().get("conv_stop", 0) or 0)

    @sel_conv_stop.setter
    def sel_conv_stop(self, value) -> None:
        self._cur()["conv_stop"] = int(value or 0)

    def set_conv_range(self, start, stop) -> None:
        """Set the convolution/fit range (from the draggable plot region)."""
        cur = self._cur()
        cur["conv_start"], cur["conv_stop"] = int(min(start, stop)), int(max(start, stop))
        self.notify("range")

    @property
    def sel_irf_start(self) -> int:
        """IRF-range start (micro-time channel) — separate from the conv range."""
        return int(self._cur().get("irf_start", 0) or 0)

    @sel_irf_start.setter
    def sel_irf_start(self, value) -> None:
        self._cur()["irf_start"] = int(value or 0)

    @property
    def sel_irf_stop(self) -> int:
        """IRF-range stop (micro-time channel) — separate from the conv range."""
        return int(self._cur().get("irf_stop", 0) or 0)

    @sel_irf_stop.setter
    def sel_irf_stop(self, value) -> None:
        self._cur()["irf_stop"] = int(value or 0)

    def set_irf_range(self, start, stop) -> None:
        """Set the IRF range (from its draggable plot region)."""
        cur = self._cur()
        cur["irf_start"], cur["irf_stop"] = int(min(start, stop)), int(max(start, stop))
        self.notify("range")

    def _hist_key(self):
        """Cache key for the current (file, detector, IRF list), or None."""
        det = self.display_detector
        if not det or det not in self.detectors or not self.filename:
            return None
        return (self.filename, det, tuple(self._cur().get("irf") or []))

    def _cached_histograms(self) -> dict | None:
        """Return the cached histograms (never blocks — None if not computed yet)."""
        key = self._hist_key()
        return self._hist_cache.get(key) if key else None

    def needs_histograms(self) -> bool:
        """Return True when the current detector's histograms still need binning."""
        key = self._hist_key()
        return key is not None and key not in self._hist_cache

    def ensure_histograms(self) -> dict | None:
        """Bin the data + raw-IRF histograms (BLOCKING — run on a background thread).

        The read + binning happens in a **worker process** (the tttrlib file read
        holds the GIL, so a background thread alone still freezes the UI). The raw
        photons are binned only when the file, detector, or IRF list changes; the
        result is cached so interactive shift/bg/region ops are cheap.
        """
        key = self._hist_key()
        if key is None:
            return None
        cached = self._hist_cache.get(key)
        if cached is not None:
            return cached

        from chisurf.core.fluorescence.imaging import calibration_histograms, detector_ps_channels

        det = self.display_detector
        ch_p, ch_s = detector_ps_channels(self.detectors[det])
        chs = self.detectors[det].get("chs", [0]) or [0]
        result = calibration_histograms(self.filename, chs, ch_p, ch_s, list(key[2]))
        self._hist_cache[key] = result
        return result

    def set_bg_range(self, start, stop) -> None:
        """Set the BG-estimation region and estimate VV/VH backgrounds from the data.

        The mean per-channel data counts in ``[start, stop)`` (parallel and
        perpendicular separately) become ``bg_vv`` / ``bg_vh`` — a cached histogram
        op (no effect until the histograms have been binned in the background).
        """
        import numpy as np

        cur = self._cur()
        s, e = int(min(start, stop)), int(max(start, stop))
        cur["bg_start"], cur["bg_stop"] = s, e
        hist = self._cached_histograms()
        if hist is not None and e > s:
            vv, vh = hist["data_vv"], hist["data_vh"]
            cur["bg_vv"] = float(np.mean(vv[s:e])) if vv[s:e].size else 0.0
            cur["bg_vh"] = float(np.mean(vh[s:e])) if vh[s:e].size else 0.0
        self.notify("run")

    def decay_data(self) -> dict | None:
        """Return the selected detector's decay + IRF + ranges for plotting.

        Uses cached histograms; only the (cheap) shift/background/normalize ops run
        on every interactive change — the raw photons are not re-binned.
        """
        det = self.display_detector
        hist = self._cached_histograms()  # non-blocking; bg thread fills the cache
        if hist is None:
            return None
        data = hist["data"]
        n = hist["n"]
        cur = self.calibration.setdefault(det, _blank())
        if not cur.get("conv_stop"):
            cur["conv_stop"] = n
        if not cur.get("irf_stop"):
            cur["irf_stop"] = n
        if not cur.get("bg_stop"):
            cur["bg_start"], cur["bg_stop"] = 0, max(1, n // 20)  # default pre-pulse region
        irf_vv = irf_vh = None
        if hist["irf_vv_raw"] is not None:
            try:
                from chisurf.core.fluorescence.imaging import prepare_irf_hist

                bg_vv = float(cur.get("bg_vv", 0.0) or 0.0)
                bg_vh = float(cur.get("bg_vh", 0.0) or 0.0)
                prepared = prepare_irf_hist(
                    hist["irf_vv_raw"], hist["irf_vh_raw"],
                    shift_vv=float(cur.get("shift_vv", 0.0) or 0.0),
                    shift_vh=float(cur.get("shift_vh", 0.0) or 0.0),
                    background_vv=(bg_vv or None), background_vh=(bg_vh or None),
                )
                irf_vv = prepared["vv"] if prepared["vv"].any() else None
                irf_vh = prepared["vh"] if prepared["vh"].any() else None
            except Exception:
                logger.debug("IRF prepare failed", exc_info=True)
        return {
            "data": data, "irf_vv": irf_vv, "irf_vh": irf_vh, "n": n,
            "conv": (int(cur.get("conv_start", 0)), int(cur.get("conv_stop", n))),
            "irf_range": (int(cur.get("irf_start", 0)), int(cur.get("irf_stop", n))),
            "bg_range": (int(cur.get("bg_start", 0)), int(cur.get("bg_stop", 0))),
            "bg_vv": float(cur.get("bg_vv", 0.0)), "bg_vh": float(cur.get("bg_vh", 0.0)),
        }

    def info_html(self) -> str:
        """Return the status/help text shown above the calibration editor."""
        detail = self.status_text or (
            "Per detector: add IRF file(s); drag the blue (conv/fit), green (IRF) "
            "and grey (background) regions on the decay — the grey region estimates "
            "VV/VH backgrounds from the data. Transferred to Phasor and MLE. Skip "
            "to use raw data."
        )
        return f"<i>{detail}</i>"

    # ── action ──
    def apply(self) -> None:
        """Publish the current calibration to the following steps."""
        if callable(self.publish):
            try:
                self.publish(dict(self.calibration))
                n = sum(
                    1 for v in self.calibration.values()
                    if v.get("irf") or v.get("bg_vv") or v.get("bg_vh")
                )
                self.status_text = f"Applied calibration ({n} detector(s) set)."
            except Exception as exc:
                self.status_text = f"Apply failed: {exc}"
        self.notify("apply")
