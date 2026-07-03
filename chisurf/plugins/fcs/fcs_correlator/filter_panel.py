"""Photon/Burst filter panel — AutoForm model and custom sections.

An AutoForm rendering of the photon/burst filter step used inside the FCS
Correlator workflow. Files come from the Files step, the container type/setup
from the Detector step, and the kept photons are consumed by the Correlator, so
this panel only exposes the filter *parameters* and the diagnostic plots — no
file-drop / setup / save toolbar (unlike the standalone ``WizardTTTRPhotonFilter``).

Only the two workflow-relevant modes are implemented here: ``burst`` and
``count_rate`` (reusing the Qt-free core algorithms in
``chisurf.core.fluorescence.burst``). The standalone widget remains for the
advanced modes (bocpd/kalman/cusum, MCS/decay histograms).
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np
from qtpy import QtWidgets

from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import register_section

_GUI_DIR = pathlib.Path(__file__).resolve().parent


def _parse_int_list(s: str) -> typing.List[int]:
    if not s:
        return []
    out: typing.List[int] = []
    for tok in str(s).replace(",", " ").split():
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


def _parse_ranges(s: str) -> typing.List[typing.Tuple[int, int]]:
    if not s:
        return []
    ranges: typing.List[typing.Tuple[int, int]] = []
    for seg in str(s).replace(",", ";").split(";"):
        seg = seg.strip()
        if not seg:
            continue
        if "-" in seg[1:]:
            a, b = seg[0] + seg[1:].split("-", 1)[0], seg[1:].split("-", 1)[1]
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            ranges.append((min(lo, hi), max(lo, hi)))
    return ranges


class FilterSettingsModel:
    """Filter parameters + selection/plot data sources for the AutoForm panel."""

    def __init__(self):
        # Channel / micro-time selection
        self.channel_numbers = ""
        self.microtime_range = "0-4095"
        # Macro-time interval (ms)
        self.min_dmt = 0.001
        self.use_min = False
        self.max_dmt = 1.500
        self.use_max = True
        # Filter
        self.filter_mode = "burst"
        self.filter_enabled = True
        self.invert = False
        self.min_ph = 10
        self.cr_tw = 5             # photon window (burst count-rate estimate)
        # Advanced mode parameters (defaults mirror the standalone widget)
        self.trace_bin_width = 0.25   # ms — binning for bocpd/kalman
        self.use_gap_fill = False
        self.max_gap = 3
        # BOCPD
        self.bocpd_prior_count = 1.0
        self.bocpd_prior_duration = 1.0
        self.bocpd_changepoint_prob = 0.45
        # Kalman
        self.kalman_q = 1.0
        self.kalman_r_scale = 1.0
        self.kalman_z_thresh = 3.0
        self.kalman_min_len = 2
        self.kalman_merge_gap = 3
        # CUSUM
        self.cusum_bg_rate = 1.0
        self.cusum_sb_ratio = 1.0
        self.cusum_alpha = 0.45
        self.cusum_beta = 20.0
        # Plot settings
        self.mcs_bin_width = 1.0   # ms
        self.range_lo = 0
        self.range_hi = 0          # 0 => full length

        self._tttr = None
        self._tttr_objects: dict = {}
        self._files: list[str] = []
        self._sel_cache_id = None
        self._sel_cache = None
        self._form: typing.Any = None
        self._last_mode = self.filter_mode
        self._refresh_timer = None

    # Which extra parameters are relevant to each filter mode (everything not
    # listed here is always shown). Used to render only the relevant controls.
    _MODE_PARAMS = {
        "burst": {"cr_tw"},
        "count_rate": set(),
        "bocpd": {
            "trace_bin_width", "use_gap_fill", "max_gap",
            "bocpd_prior_count", "bocpd_prior_duration", "bocpd_changepoint_prob",
        },
        "kalman": {
            "trace_bin_width", "use_gap_fill", "max_gap",
            "kalman_q", "kalman_r_scale", "kalman_z_thresh",
            "kalman_min_len", "kalman_merge_gap",
        },
        "cusum": {
            "use_gap_fill", "max_gap",
            "cusum_bg_rate", "cusum_sb_ratio", "cusum_alpha", "cusum_beta",
        },
    }

    def view_spec(self):
        import copy
        import json

        with open(_GUI_DIR / "filter.view.json", encoding="utf-8") as fh:
            data = json.load(fh)
        conditional = set().union(*self._MODE_PARAMS.values())
        allowed = self._MODE_PARAMS.get(self.filter_mode, set())

        def prune(sections):
            out = []
            for sec in sections:
                if "sections" in sec:
                    sec = dict(sec)
                    sec["sections"] = prune(sec["sections"])
                    # drop panels that end up empty after pruning
                    if not sec["sections"]:
                        continue
                attr = sec.get("attr")
                if attr in conditional and attr not in allowed:
                    continue
                # Auto-update: every bound field notifies the model on change.
                if sec.get("type") in ("value", "choice", "toggle") and attr:
                    sec = dict(sec)
                    sec["call"] = "on_param_changed"
                out.append(sec)
            return out

        data = copy.deepcopy(data)
        data["sections"] = prune(data["sections"])
        return load_view_spec(data)

    # -- live update ---------------------------------------------------------

    def on_param_changed(self, value=None) -> None:
        """Called by AutoForm whenever a bound field changes (auto-update)."""
        if self.filter_mode != self._last_mode:
            # Mode switched: re-render so only the relevant controls show.
            self._last_mode = self.filter_mode
            self._sel_cache_id = None
            form = self._form
            if form is not None:
                from qtpy import QtCore
                QtCore.QTimer.singleShot(0, form.rebuild)
            return
        self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        """Debounce recomputes so dragging/typing stays responsive."""
        from qtpy import QtCore

        self._sel_cache_id = None
        if self._refresh_timer is None:
            self._refresh_timer = QtCore.QTimer()
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.timeout.connect(self._do_refresh)
        self._refresh_timer.start(150)

    def _do_refresh(self) -> None:
        if self._form is not None:
            self._form.refresh_plots()

    # -- data plumbing -------------------------------------------------------

    def set_tttr_objects(self, objs: dict, files: list[str]) -> None:
        self._tttr_objects = objs or {}
        self._files = list(files or [])
        self._tttr = None
        for f in self._files:
            tt = self._tttr_objects.get(str(pathlib.Path(f).resolve()))
            if tt is not None:
                self._tttr = tt
                break
        self._sel_cache_id = None
        if self.range_hi <= 0 and self._tttr is not None:
            self.range_hi = len(self._tttr)
        if self._form is not None:
            self._form.refresh_plots()

    # -- selection -----------------------------------------------------------

    @staticmethod
    def _dT(tttr) -> np.ndarray:
        mt = np.asarray(tttr.macro_times)
        d = np.diff(mt, prepend=mt[0]).astype(float)
        return d * tttr.header.macro_time_resolution * 1000.0

    def compute_selection(self, tttr) -> np.ndarray:
        """Boolean keep-mask for *tttr* under the current parameters."""
        import tttrlib
        from chisurf.core.fluorescence import burst as burstmod

        n = len(tttr)
        s = np.ones(n, dtype=bool)

        chs = _parse_int_list(self.channel_numbers)
        if chs:
            m = tttrlib.TTTRMask()
            m.select_channels(tttr, chs, mask=True)
            s &= np.asarray(m.get_mask()).astype(bool)

        mtr = _parse_ranges(self.microtime_range)
        if mtr and mtr != [(0, 4095)]:
            m = tttrlib.TTTRMask()
            m.select_microtime_ranges(tttr, mtr)
            m.flip()
            s &= np.asarray(m.get_mask()).astype(bool)

        dT = self._dT(tttr)
        if self.use_min:
            s &= dT >= float(self.min_dmt)
        if self.use_max:
            s &= dT <= float(self.max_dmt)

        if self.filter_enabled:
            mode = self.filter_mode
            tw = float(self.max_dmt) / 1000.0
            if mode == "count_rate":
                idx = burstmod.count_rate_filter(
                    tttr=tttr,
                    n_ph_max=int(self.min_ph),
                    time_window=tw,
                    invert=bool(self.invert),
                    make_mask=True,
                )
                s &= np.asarray(idx) >= 0
            elif mode == "burst":
                sel = np.asarray(
                    burstmod.burst_filter(
                        tttr=tttr,
                        min_ph=int(self.min_ph),
                        ph_window=int(self.cr_tw),
                        time_window=tw,
                    )
                ).astype(bool)
                s &= sel
            elif mode in ("bocpd", "kalman"):
                s &= self._changepoint_selection(tttr, mode)
            elif mode == "cusum":
                import chisurf.core.fluorescence.burst.cusum as cusum_mod
                sel = np.asarray(
                    cusum_mod.cusum_filter(
                        tttr=tttr,
                        min_ph=int(self.min_ph),
                        background_rate=int(self.cusum_bg_rate),
                        sb_ratio=float(self.cusum_sb_ratio),
                        alpha=float(self.cusum_alpha),
                        beta=float(self.cusum_beta),
                    )
                ).astype(bool)
                s &= sel

            # ``count_rate`` handles inversion internally; invert the others here.
            if self.invert and mode != "count_rate":
                s = ~s

        if self.use_gap_fill and int(self.max_gap) > 0:
            from chisurf.core.math.signal import fill_small_gaps_in_array
            s = fill_small_gaps_in_array(s, max_gap=int(self.max_gap))
        return s

    def _changepoint_selection(self, tttr, mode: str) -> np.ndarray:
        """BOCPD / Kalman burst detection -> per-photon keep mask."""
        from chisurf.core.fluorescence import burst as burstmod
        from chisurf.core.fluorescence.burst.utils import create_array_with_ones

        n = len(tttr)
        channel_list = _parse_int_list(self.channel_numbers)
        if not channel_list:
            channel_list = list(tttr.get_used_routing_channels())
        time_unit = tttr.header.macro_time_resolution
        timestamps = np.asarray(tttr.macro_times) * time_unit
        channels = np.asarray(tttr.routing_channels)
        ts_list = []
        for ch in channel_list:
            cts = timestamps[channels == ch]
            if len(cts) == 0:
                return np.ones(n, dtype=bool)
            ts_list.append(cts)

        dt = float(self.trace_bin_width) / 1000.0
        if mode == "bocpd":
            bursts, *_ = burstmod.bocpd_burst_detection_multi(
                ts_list,
                dt=dt,
                prior_count=float(self.bocpd_prior_count),
                prior_duration=float(self.bocpd_prior_duration),
                changepoint_prob=float(self.bocpd_changepoint_prob),
                max_run=256,
                min_counts=int(self.min_ph),
            )
            start_stop = burstmod.bocpd.convert_bursts_to_start_stop(bursts, tttr)
        else:  # kalman
            bursts, *_ = burstmod.kalman_burst_detection_multi(
                ts_list,
                dt=dt,
                q=float(self.kalman_q),
                r_scale=float(self.kalman_r_scale),
                z_thresh=float(self.kalman_z_thresh),
                min_len=int(self.kalman_min_len),
                merge_gap=int(self.kalman_merge_gap),
                min_counts=int(self.min_ph),
            )
            start_stop = burstmod.kalman.convert_bursts_to_start_stop(bursts, tttr)

        if len(start_stop) == 0:
            return np.ones(n, dtype=bool)
        return np.asarray(create_array_with_ones(start_stop, n)).astype(bool)

    def selected(self) -> np.ndarray | None:
        if self._tttr is None:
            return None
        key = id(self._tttr)
        if self._sel_cache_id != key:
            self._sel_cache = self.compute_selection(self._tttr)
            self._sel_cache_id = key
        return self._sel_cache

    def invalidate(self) -> None:
        self._sel_cache_id = None
        if self._form is not None:
            self._form.refresh_plots()

    # -- plot data sources ---------------------------------------------------

    def dt_scatter_series(self):
        if self._tttr is None:
            return []
        dT = self._dT(self._tttr)
        sel = self.selected()
        if sel is None:
            return []
        lo = max(0, int(self.range_lo))
        hi = int(self.range_hi) if self.range_hi > 0 else len(dT)
        hi = min(hi, len(dT))
        if hi <= lo:
            return []
        x = np.arange(lo, hi)
        y = dT[lo:hi]
        m = sel[lo:hi].astype(bool)
        return [
            {"x": x[m], "y": y[m], "name": "selected", "color": "c"},
            {"x": x[~m], "y": y[~m], "name": "removed", "color": "y"},
        ]

    def count_rate_series(self):
        if self._tttr is None:
            return []
        tw = max(1e-6, float(self.mcs_bin_width) / 1000.0)
        try:
            trace_all = np.asarray(self._tttr.get_intensity_trace(time_window_length=tw))
        except Exception:
            return []
        x = np.arange(len(trace_all)) * float(self.mcs_bin_width) / 1000.0
        series = [{"x": x, "y": trace_all / (tw * 1000.0), "name": "all", "color": "y"}]
        sel = self.selected()
        if sel is not None and sel.any():
            try:
                idx = np.where(sel)[0]
                tt_sel = self._tttr[idx]
                trace_sel = np.asarray(tt_sel.get_intensity_trace(time_window_length=tw))
                xs = np.arange(len(trace_sel)) * float(self.mcs_bin_width) / 1000.0
                series.append(
                    {"x": xs, "y": trace_sel / (tw * 1000.0), "name": "selected", "color": "c"}
                )
            except Exception:
                pass
        return series

    def info_text(self) -> str:
        if self._tttr is None:
            return "No data loaded."
        sel = self.selected()
        if sel is None:
            return "No data loaded."
        n = len(sel)
        kept = int(sel.sum())
        pct = 100.0 * kept / n if n else 0.0
        return f"{kept:,} / {n:,} photons kept ({pct:.1f}%)"


# ---- Custom AutoForm sections ----------------------------------------------


class _FilterPlot(QtWidgets.QWidget):
    """Reusable plot that redraws a model source on refresh."""

    AUTOFORM_REFRESH = True

    def __init__(self, model, source, title, *, log_y=False, x_label="", y_label=""):
        super().__init__()
        import pyqtgraph as pg

        self._model = model
        self._source = source
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setTitle(title)
        if x_label:
            self.plot.setLabel("bottom", x_label)
        if y_label:
            self.plot.setLabel("left", y_label)
        if log_y:
            try:
                self.plot.setLogMode(False, True)
            except Exception:
                pass
        try:
            self.plot.getPlotItem().getViewBox().setMenuEnabled(False)
        except Exception:
            pass
        layout.addWidget(self.plot)
        self.refresh()

    def refresh(self):
        import pyqtgraph as pg

        src = getattr(self._model, self._source, None)
        if not callable(src):
            return
        try:
            series = src() or []
        except Exception:
            return
        self.plot.clear()
        scatter = self._source == "dt_scatter_series"
        for s in series:
            x = np.asarray(s.get("x", []))
            y = np.asarray(s.get("y", []))
            if scatter:
                self.plot.plot(
                    x, y, pen=None, symbol="o", symbolSize=2,
                    symbolBrush=s.get("color", "w"), symbolPen=None, name=s.get("name", ""),
                )
            else:
                self.plot.plot(x, y, pen=pg.mkPen(s.get("color", "w"), width=1), name=s.get("name", ""))


@register_section("filter_dt_plot")
class _FilterDtPlot(_FilterPlot):
    """dT scatter with a draggable horizontal region bound to min/max dMT."""

    _region = None

    def __init__(self, model, target: str = "", **options):
        super().__init__(
            model, "dt_scatter_series", "Delta macro-time",
            log_y=True, x_label="Photon index", y_label="dT (ms)",
        )

    def refresh(self):
        super().refresh()
        import pyqtgraph as pg

        if self._region is None:
            self._region = pg.LinearRegionItem(
                orientation="horizontal", brush=(80, 180, 255, 40),
            )
            self._region.sigRegionChangeFinished.connect(self._on_region)
        # ``clear()`` in the base refresh removed the region; re-add and position
        # it from the model (log-y axis => region values are log10).
        self.plot.addItem(self._region)
        lo = max(float(self._model.min_dmt), 1e-12)
        hi = max(float(self._model.max_dmt), lo * (1.0 + 1e-6))
        self._region.blockSignals(True)
        self._region.setRegion((np.log10(lo), np.log10(hi)))
        self._region.blockSignals(False)

    def _on_region(self):
        lo, hi = self._region.getRegion()
        a, b = 10.0 ** lo, 10.0 ** hi
        self._model.min_dmt = float(min(a, b))
        self._model.max_dmt = float(max(a, b))
        self._model.use_min = True
        self._model.use_max = True
        form = getattr(self._model, "_form", None)
        if form is not None:
            try:
                form.sync_fields()
            except Exception:
                pass
        self._model.on_param_changed()


@register_section("filter_cr_plot")
class _FilterCrPlot(_FilterPlot):
    def __init__(self, model, target: str = "", **options):
        super().__init__(
            model, "count_rate_series", "Count rate",
            x_label="Time (s)", y_label="Intensity (kHz)",
        )


@register_section("filter_info")
class _FilterInfo(QtWidgets.QWidget):
    AUTOFORM_REFRESH = True

    def __init__(self, model, target: str = "", **options):
        super().__init__()
        self._model = model
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel("No data loaded.")
        layout.addWidget(self.label)
        layout.addStretch(1)

    def refresh(self) -> None:
        self.label.setText(self._model.info_text())
