from __future__ import annotations

import numpy as np

from .qt_stack import ensure_qt_stack


class _MaxentPlottingMixin:
    def _update_plots_from_result(
        self,
        decay: np.ndarray,
        t: np.ndarray,
        result: dict,
    ) -> None:
        pg, _, QtCore, _, _ = ensure_qt_stack()

        if "R" in result:
            dist_axis = np.asarray(result["R"], dtype=float).ravel()
            self.plot_dist.setLabel("bottom", "distance", units="\u00c5")
            self.plot_dist.setTitle("Distance distribution")
        else:
            dist_axis = np.asarray(result["tau"], dtype=float).ravel()
            self.plot_dist.setLabel("bottom", "lifetime", units="ns")
            self.plot_dist.setTitle("Lifetime distribution")

        p = np.asarray(result["p"], dtype=float).ravel()
        Fi = np.asarray(result["Fi"], dtype=float)
        y_seg = np.asarray(result["y"], dtype=float).ravel()
        sigma = np.asarray(result["sigma"], dtype=float).ravel()
        fitstart, fitstop = result["fitrange"]

        s = float(p.sum())
        if s > 0.0:
            p_norm = p / s
        else:
            p_norm = p

        fit_seg = (Fi @ p) * sigma
        try:
            fit_add = np.asarray(result.get("fit_additive", []), dtype=float).ravel()
        except Exception:
            fit_add = np.zeros(0, dtype=float)
        if fit_add.size == fit_seg.size:
            fit_seg = fit_seg + fit_add
        wres = (y_seg - fit_seg) / sigma

        t_seg = t[fitstart : fitstop + 1]

        self.plot_decay.clear()
        if getattr(self, "_fit_region", None) is not None:
            self.plot_decay.addItem(self._fit_region)
        decay_plot = np.maximum(decay, 1.0)
        fit_plot = np.zeros_like(decay_plot)
        fit_plot[fitstart : fitstop + 1] = np.maximum(fit_seg, 1.0)

        self.plot_decay.plot(t, decay_plot, pen="w", name="data")
        self.plot_decay.plot(t, fit_plot, pen="y", name="MEM fit")

        self.plot_wres.clear()
        self.plot_wres.plot(t_seg, wres, pen="c")
        self.plot_wres.addLine(y=0.0, pen=pg.mkPen("w", width=1))

        self.plot_dist.clear()
        self._sample_band_lower = None
        self._sample_band_upper = None
        self._sample_band_fill = None
        self._sample_hist_item = None

        try:
            prior = np.asarray(result.get("prior", []), dtype=float).ravel()
        except Exception:
            prior = np.zeros(0, dtype=float)
        if prior.size == dist_axis.size:
            sp = float(np.sum(prior))
            if sp > 0.0:
                prior_n = prior / sp
            else:
                prior_n = prior
            try:
                self.plot_dist.plot(
                    dist_axis,
                    prior_n,
                    pen=pg.mkPen((180, 180, 180, 200), style=QtCore.Qt.DashLine),
                )
            except Exception:
                pass

        self.plot_dist.plot(dist_axis, p_norm, pen="m", symbol="o", symbolSize=4)

        stats = getattr(self, "_sample_stats", None)
        if stats is None:
            return

        try:
            axis_s = np.asarray(stats.get("axis", []), dtype=float).ravel()
        except Exception:
            return

        if axis_s.size != dist_axis.size:
            return
        try:
            if not np.allclose(axis_s, dist_axis):
                return
        except Exception:
            return

        try:
            p_lo = np.asarray(stats.get("p_lo", []), dtype=float).ravel()
            p_hi = np.asarray(stats.get("p_hi", []), dtype=float).ravel()
            p_med = np.asarray(stats.get("p_med", []), dtype=float).ravel()
            p_mean = np.asarray(stats.get("p_mean", []), dtype=float).ravel()
            p_mem_s = np.asarray(stats.get("p_mem", []), dtype=float).ravel()
        except Exception:
            return
        if p_lo.size != dist_axis.size or p_hi.size != dist_axis.size:
            return

        def _norm_prob(v: np.ndarray) -> np.ndarray:
            try:
                s = float(np.sum(v))
            except Exception:
                s = 0.0
            if not np.isfinite(s) or s <= 0.0:
                return v
            return v / s

        p_lo_n = _norm_prob(p_lo)
        p_hi_n = _norm_prob(p_hi)
        if p_med.size == dist_axis.size:
            p_med_n = _norm_prob(p_med)
        elif p_mean.size == dist_axis.size:
            p_med_n = _norm_prob(p_mean)
        else:
            p_med_n = p_lo_n * 0.0

        lower_curve = self.plot_dist.plot(
            dist_axis,
            p_lo_n,
            pen=pg.mkPen((80, 80, 200, 160)),
        )
        upper_curve = self.plot_dist.plot(
            dist_axis,
            p_hi_n,
            pen=pg.mkPen((80, 80, 200, 160)),
        )

        fill_item = None
        try:
            fill_item = pg.FillBetweenItem(upper_curve, lower_curve, brush=(80, 80, 200, 80))
            self.plot_dist.addItem(fill_item)
        except Exception:
            fill_item = None

        self._sample_band_lower = lower_curve
        self._sample_band_upper = upper_curve
        self._sample_band_fill = fill_item

        try:
            if dist_axis.size > 1:
                dx = float(np.median(np.diff(dist_axis)))
            else:
                dx = 1.0
            width = 0.9 * dx
            hist_item = pg.BarGraphItem(
                x=dist_axis,
                height=p_med_n,
                width=width,
                brush=(100, 150, 255, 80),
                pen=None,
            )
            self.plot_dist.addItem(hist_item)
            self._sample_hist_item = hist_item
        except Exception:
            self._sample_hist_item = None
